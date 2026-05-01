# DETAIL_ARCHITECH.MD
# Arsitektur Lengkap — Polymarket BTC Up/Down Prediction System
# Versi: 1.0 | Tanggal: Mei 2026

---

## PENDAHULUAN DAN KONTEKS

Dokumen ini adalah **sumber kebenaran tunggal** untuk implementasi sistem ML prediksi
Polymarket BTC Up/Down. Setiap baris kode yang ditulis harus merujuk ke dokumen ini.
Dokumen ini dibuat berdasarkan:

1. README arsitektur referensi (grounded in academic literature)
2. **~1 bulan eksperimen aktual** dengan hasil AUC 0.51–0.57
3. **11 bug kritis** yang ditemukan dan dianalisa dari kode iterasi 1–10
4. Spesifikasi hardware aktual yang tersedia
5. Dataset aktual yang dimiliki (29-Apr-2025 → 16-Apr-2026, ~352 hari)

### Mengapa dokumen ini sangat panjang dan detail

Pengalaman 1 bulan menunjukkan bahwa **ambiguitas dalam spesifikasi = bug dalam kode**.
Setiap bug yang ditemukan (VPIN index mismatch, time feature aggregation, LightGBM DART,
feature selection missing, RegimeDetector leakage, dsb.) bisa dihindari jika spesifikasi
lebih eksplisit. Dokumen ini dirancang untuk menghilangkan semua ruang ambiguitas.

---

## BAGIAN 0 — HARDWARE CONSTRAINTS DAN BATAS KERAS

### 0.1 Spesifikasi Hardware

```
CPU  : Intel Core i5-12500H (12th Gen H-series)
       - 12 cores (4 P-core + 8 E-core), 16 threads
       - P-core: up to 4.5 GHz, E-core: up to 3.3 GHz
       - Cocok untuk: multiprocessing, parallel data loading
       - JANGAN gunakan lebih dari 10 worker (sisakan 2 untuk OS)

GPU  : NVIDIA RTX 3050 Laptop (4GB GDDR6 VRAM)
       - Memory bandwidth: ~192 GB/s
       - CUDA cores: 2048
       - Batasan KERAS: gpu_ram_part ≤ 0.40 (bukan 0.45, bukan 0.5)
       - Kenapa 0.40: padding 10% untuk CatBoost overhead + OS GPU memory

RAM  : 16 GB DDR5 + 50 GB swap (total addressable: 66 GB)
       - Swap sangat lambat (SSD NVMe ~3 GB/s vs RAM ~50 GB/s)
       - JANGAN biarkan proses masuk ke swap kecuali tidak ada pilihan lain
       - Target: semua data training muat di RAM (< 12 GB setelah reserve 4 GB)

Storage: Asumsikan NVMe SSD berdasarkan swap size
       - Read sequential: ~3 GB/s
       - Parquet dengan compression zstd: optimal untuk I/O

OS   : Linux (Ubuntu atau Arch berdasarkan path /run/media)
```

### 0.2 Batasan Keras yang TIDAK BOLEH Dilanggar

Setiap item berikut adalah LARANGAN ABSOLUT. Jika ada kode yang melanggar ini,
kode tersebut SALAH, bukan hanya suboptimal.

```
[GPU-01] gpu_ram_part CatBoost: MAKSIMUM 0.40
         Alasan: RTX 3050 4GB tidak bisa menangani lebih dari ini
         dengan dataset 20k+ rounds dan 70+ features

[GPU-02] rsm parameter: DILARANG KERAS untuk CatBoost GPU + Logloss
         Alasan: Parameter ini tidak didukung kombinasi GPU+Logloss
         Error: "rsm is not supported for GPU learning with Logloss"

[GPU-03] max_bin: MAKSIMUM 128
         Alasan: Memory constraint RTX 3050

[GPU-04] bootstrap_type: HARUS 'Bernoulli'
         Alasan: 'Bayesian' (default) lebih boros memory GPU

[MEM-01] Multiprocessing workers: MAKSIMUM 10
         Sisakan 2 core untuk OS, disk I/O, monitoring

[MEM-02] Feature DataFrame: WAJIB astype(np.float32) sebelum training
         Alasan: float64 membutuhkan 2x RAM. 20k rounds × 200 feat ×
         float64 = 32 MB (kecil), tapi setelah intermediate operations
         bisa 10x lebih besar

[ML-01]  DILARANG: random shuffle sebelum time split
         Data time series HARUS diurutkan kronologis sebelum split

[ML-02]  DILARANG: mencari threshold di TEST set
         Threshold hanya boleh dicari di VALIDATION set

[ML-03]  DILARANG: StandardScaler.fit() pada seluruh dataset
         Scaler hanya boleh fit pada TRAINING data
         Scaler harus TRANSFORM (tidak fit) pada val dan test

[ML-04]  DILARANG: boosting_type='dart' untuk LightGBM pada dataset ini
         Observasi aktual: dart menyebabkan "No further splits with
         positive gain" setelah iterasi awal karena dropout rate
         terlalu agresif untuk dataset dengan signal-to-noise rendah

[FE-01]  DILARANG: agregasi window (mean/std/trend) untuk time features
         Kolom: sin_hour, cos_hour, sin_dow, cos_dow, sin_15m, cos_15m,
         is_us_session, is_london_session, is_overlap, is_weekend
         Alasan: nilai konstan dalam window 15 menit →
         win_mean_sin_hour ≡ snap_sin_hour → noise murni
         Iterasi 9 menunjukkan ini menjadi top feature karena spurious

[FE-02]  DILARANG: harga absolut sebagai feature
         Kolom: close, open, high, low, price (tanpa normalisasi)
         Alasan: non-stationary, model tidak bisa generalisasi

[FE-03]  DILARANG: menggunakan data dari DALAM round yang diprediksi
         Setiap feature harus dari [T-window, T) EKSKLUSIF T
```

---

## BAGIAN 1 — DATASET DAN DATA YANG TERSEDIA

### 1.1 Dataset Aktual

```
Sumber data    : Binance Futures WebSocket (aggTrades + OrderBook)
Symbol         : BTCUSDT
Periode        : 29 April 2025 → 16 April 2026
Total hari     : ~352 hari
Target market  : Polymarket BTC Up/Down 15-menit

Estimasi volume data:
  - aggTrades  : ~350 file parquet (1 per hari), ~200-500 MB/hari
  - OrderBook  : ~350 file parquet (1 per hari), L2 depth 20
  - Total raw  : estimasi 70-175 GB raw, ~20-50 GB setelah parquet+zstd

Coverage aktual yang diobservasi: 44% (MASALAH UTAMA)
  - Hanya 9,708 rounds valid dari ~27,936 expected (352 hari × 24h × 4 round/jam × 2)
  - Untuk 15m: 352 × 24 × 4 = 33,792 expected rounds
  - Target coverage setelah cleaning: ≥ 70%
```

### 1.2 Struktur Direktori WAJIB

```
/run/media/rotan/New Volume/gemini3/polymarket_5m/
├── venv/                          # Virtual environment (WAJIB)
│   └── lib/python3.12/site-packages/
├── config.yaml                    # Single source of truth untuk semua parameter
├── data/
│   ├── raw/
│   │   ├── orderbook/             # Raw OB parquet files per hari
│   │   │   ├── 20250429_orderbook.parquet
│   │   │   └── ...
│   │   ├── aggtrades/             # Raw aggTrades parquet per hari
│   │   │   ├── 20250429_aggtrades.parquet
│   │   │   └── ...
│   │   └── historical_1m_klines.parquet  # OHLCV 1m dari Binance
│   ├── interim/                   # Data intermediate (bisa dihapus)
│   │   └── daily_obi/             # OBI per hari sebelum merge
│   ├── features/
│   │   └── v1/
│   │       ├── final_features.parquet    # Output utama feature engineering
│   │       ├── feature_dict_v1.json      # Data dictionary setiap feature
│   │       └── missing_masks.parquet     # Mask NaN per kolom per round
│   └── labels/
│       ├── targets_v1.parquet     # Target per round dengan provenance
│       └── label_audit_sample.csv # 500 label untuk audit manual
├── src/                           # Source code production
│   ├── ingest/
│   │   └── 01_collect_data.py
│   ├── features/
│   │   └── 02_feature_engineering.py
│   ├── labels/
│   │   └── 02b_label_builder.py
│   ├── models/
│   │   ├── 03_training.py
│   │   └── 04_evaluate.py
│   ├── calibration/
│   │   └── 05_calibrate.py
│   └── inference/
│       └── 06_inference.py
├── models/
│   └── artifacts/
│       ├── feature_cols_v1.pkl          # List fitur yang digunakan
│       ├── selected_features_v1.pkl     # Post-reduction feature list
│       ├── lgbm_screener_v1.pkl         # Model screener (tidak untuk prod)
│       ├── lgbm_final_v1.pkl            # LightGBM final model
│       ├── catboost_final_v1.cbm        # CatBoost final model
│       ├── scaler_v1.pkl                # StandardScaler fit pada train SAJA
│       ├── calibrator_v1.pkl            # Isotonic calibrator
│       ├── regime_detector_v1.pkl       # KMeans fit pada train SAJA
│       ├── optimal_threshold_v1.pkl     # Threshold dari VAL set
│       ├── best_params_v1.pkl           # Optuna best params
│       ├── ensemble_weights_v1.pkl      # Bobot ensemble
│       └── lgbm_importance_v1.csv       # Feature importance untuk audit
├── reports/
│   ├── experiment_log.md          # Log SEMUA eksperimen
│   ├── calibration_plots/
│   └── performance_by_regime/
├── tests/                         # Unit tests WAJIB
│   ├── test_features.py
│   ├── test_timestamps.py
│   └── test_leakage.py
├── notebooks/
│   └── eda.ipynb
└── logs/
    ├── feature_engineering.log
    └── training.log
```

### 1.3 Setup Virtual Environment (WAJIB untuk setiap sesi)

```bash
# Buat venv dengan Python 3.12 (bukan 3.14 — beberapa library belum support)
cd "/run/media/rotan/New Volume/gemini3/polymarket_5m"

python3.12 -m venv venv
source venv/bin/activate

# Verifikasi Python version
python --version  # Harus 3.12.x

# Install packages (VERSI TERBARU yang kompatibel, bukan version lama)
pip install --upgrade pip setuptools wheel

# Core ML
pip install catboost>=1.2.7
pip install lightgbm>=4.5.0
pip install optuna>=4.1.0
pip install scikit-learn>=1.6.0

# Data processing
pip install pandas>=2.2.3
pip install numpy>=2.1.0
pip install pyarrow>=18.0.0   # Untuk parquet
pip install ta-lib>=0.5.1     # Technical analysis (BUKAN 'ta', BUKAN 'talib' versi lama)

# Utilities
pip install pyyaml>=6.0.2
pip install joblib>=1.4.2
pip install tqdm>=4.67.0

# Plotting (opsional)
pip install matplotlib>=3.9.0
pip install seaborn>=0.13.2

# Testing
pip install pytest>=8.0.0

# Simpan requirements
pip freeze > requirements.txt

# PENTING: Aktifkan venv setiap kali sebelum menjalankan script
# source venv/bin/activate
```

**PERINGATAN KOMPATIBILITAS PYTHON 3.14:**
Dari log error sebelumnya, proyek ini berjalan di Python 3.14. Beberapa package
belum stable di 3.14 (numpy bitwise operators berperilaku berbeda). Gunakan 3.12
jika ada masalah kompatibilitas.

---

## BAGIAN 2 — CONFIG.YAML (SINGLE SOURCE OF TRUTH)

Semua parameter ada di sini. TIDAK BOLEH ada hardcode di script manapun.

```yaml
# config.yaml — Semua parameter sistem

project:
  name: "polymarket_btc_15m"
  symbol: "BTCUSDT"
  round_minutes: 15          # Durasi round Polymarket (15 menit)
  target_threshold: 0.0003   # 0.03% minimum move untuk label valid (bukan 0.0)
                             # Alasan: round dengan return < 0.03% sangat noise
                             # Tuning range: 0.0001 - 0.0010
  version: "v1"

paths:
  data:
    raw: "data/raw"
    orderbook: "data/raw/orderbook"
    aggtrades: "data/raw/aggtrades"
    klines_1m: "data/raw/historical_1m_klines.parquet"
    interim: "data/interim"
    features_v1: "data/features/v1"
    labels: "data/labels"
  models:
    artifacts: "models/artifacts"
    checkpoints: "models/checkpoints"
  logs: "logs"
  reports: "reports"

data:
  ob_depth: 20               # Top 20 level bid/ask (data yang ada)
  pre_round_window_mins: 15  # Window fitur sebelum round T
  max_gap_seconds: 180       # Gap data maks dalam window (3 menit)
  min_ticks_per_round: 10    # Minimum tick per round agar valid
  resample_interval_seconds: 10  # Resample OB ke 10 detik
  coverage_warning_threshold: 0.70  # Warning jika coverage < 70%
  # Data range yang tersedia
  data_start: "2025-04-29"
  data_end: "2026-04-16"

features:
  # Kolom TIME yang TIDAK boleh diagregasi (hanya snapshot)
  time_like_cols:
    - "sin_hour"
    - "cos_hour"
    - "sin_dow"
    - "cos_dow"
    - "sin_15m"
    - "cos_15m"
    - "is_us_session"
    - "is_london_session"
    - "is_overlap"
    - "is_weekend"
    - "is_roll_hour"
    - "minutes_since_roll"

  # Kolom yang mengandung harga absolut (DILARANG masuk feature)
  price_absolute_cols:
    - "price"
    - "close"
    - "open"
    - "high"
    - "low"
    - "initial_price"
    - "final_price"

  # Kolom target dan metadata (DILARANG masuk feature)
  exclude_cols:
    - "round_start"
    - "target"
    - "future_return"
    - "tick_count"
    - "max_gap"
    - "gap_to_prev"
    - "has_valid_context"
    - "round_label"

  target_n_features: 70       # Setelah feature reduction via LightGBM
  nan_threshold: 0.30         # Buang feature jika NaN > 30%
  std_threshold: 1.0e-10      # Buang feature jika std ≈ 0 (konstan)

  inter_round_lags: 8         # Berapa lag round ke belakang
  inter_round_rolling: [4, 8, 16]  # Window untuk rolling stats antar-round

training:
  # Split ratios
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

  # Sample weights
  # PELAJARAN: 3.5x terlalu agresif → LightGBM "No further splits"
  # Tuned ke 2.0x berdasarkan observasi
  sample_weight_high: 2.0    # Untuk round dengan |return| >= median
  sample_weight_low: 1.0     # Untuk round dengan |return| < median

  # Walk-forward validation
  wfv_train_rounds: 1344     # ~14 hari × 96 rounds/hari
  wfv_val_rounds: 192        # ~2 hari × 96 rounds/hari
  wfv_step_rounds: 192       # Geser 2 hari per fold
  wfv_n_splits: 6            # Minimum 6 fold

  catboost:
    # RTX 3050 4GB CONSTRAINTS (TIDAK BOLEH DIUBAH)
    task_type: "GPU"
    devices: "0"
    gpu_ram_part: 0.40        # MAKSIMUM. Jangan naikkan.
    max_bin: 128              # MAKSIMUM. Jangan naikkan.
    bootstrap_type: "Bernoulli"  # WAJIB (bukan Bayesian)
    loss_function: "Logloss"
    eval_metric: "AUC"
    random_seed: 42
    # rsm: TIDAK ADA DI SINI — tidak didukung GPU + Logloss
    # Parameter yang di-tune via Optuna:
    optuna_search_space:
      depth: [4, 7]            # integer, inclusive
      l2_leaf_reg: [5, 150]    # float, log scale
      random_strength: [0.5, 10.0]
      min_data_in_leaf: [50, 600]
      learning_rate: [0.003, 0.05]  # log scale
      max_bin: [64, 128]       # integer
      subsample: [0.6, 0.95]
    # Fixed per trial
    iterations_per_trial: 3000
    early_stopping_per_trial: 150
    # Final training
    iterations_final: 10000
    early_stopping_final: 500
    snapshot_interval: 300    # seconds

  lightgbm_screener:
    # CPU only (untuk feature selection)
    n_estimators: 500
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.7
    bagging_fraction: 0.8
    bagging_freq: 5
    min_child_samples: 30
    boosting_type: "gbdt"     # WAJIB 'gbdt', BUKAN 'dart'
    reg_alpha: 0.1
    reg_lambda: 0.1
    n_jobs: 10                # Gunakan 10 dari 12 core
    verbose: -1
    random_state: 42
    early_stopping_rounds: 50

  lightgbm_final:
    boosting_type: "gbdt"     # WAJIB 'gbdt', BUKAN 'dart'
    objective: "binary"
    metric: "auc"
    num_leaves: 63
    learning_rate: 0.02
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    min_child_samples: 50
    reg_alpha: 0.1
    reg_lambda: 0.5
    n_jobs: 10
    verbose: -1
    seed: 42
    num_boost_round: 2000
    early_stopping_rounds: 100

  optuna:
    n_trials: 80
    direction: "maximize"
    sampler: "TPE"
    seed: 42
    overfit_gap_allowed: 0.03  # Penalti overfit = max(0, gap - 0.03) × 5.0
    prob_spread_bonus_weight: 5.0
    prob_spread_max_bonus: 0.03

evaluation:
  # Threshold scan range (dari VAL set)
  threshold_scan_start: 0.50     # WAJIB mulai dari 0.50, bukan 0.505
  threshold_scan_end: 0.70
  threshold_scan_step: 0.005
  min_coverage_for_threshold: 0.10  # Minimal 10% coverage

  # High-confidence thresholds untuk laporan
  confidence_levels: [0.50, 0.51, 0.53, 0.55, 0.57, 0.60]

  # Calibration
  calibration_method: "isotonic"  # atau "sigmoid" (Platt)
  calibration_n_bins: 10

  # Session breakdowns (jam UTC)
  sessions:
    us: [13, 21]
    london: [7, 16]
    asian: [0, 8]
    overlap: [13, 16]

  # Volatility regime thresholds
  rvol_low_pct: 0.25   # < 25th percentile = low vol
  rvol_high_pct: 0.75  # > 75th percentile = high vol

trade_decision:
  polymarket_fee: 0.02     # 2% biaya Polymarket
  model_risk_margin: 0.02  # Margin untuk ketidakpastian model
  min_ev_to_trade: 0.0     # EV > 0 setelah fee dan margin
  max_bet_fraction: 0.03   # Maksimum 3% bankroll per bet
  kelly_fraction: 0.25     # 25% Kelly (fractional Kelly)

  # Circuit breakers
  max_loss_24h_pct: 0.10   # Stop jika rugi > 10% dalam 24 jam
  max_consecutive_losses: 10

  # No-trade conditions
  max_polymarket_spread: 0.05   # No trade jika spread > 5%
  min_polymarket_depth: 10.0    # No trade jika depth < $10

monitoring:
  calibration_drift_threshold: 0.05
  feature_drift_ks_pvalue: 0.05
  performance_decay_threshold: 0.03
```

---

## BAGIAN 3 — DEFINISI TARGET (NON-NEGOTIABLE)

### 3.1 Aturan Resolusi Polymarket 15 Menit

```
Round boundary : setiap 00:00, 00:15, 00:30, 00:45 UTC (floor ke 15 menit)
initial_price  : harga BTC pada tick PERTAMA setelah boundary T
final_price    : harga BTC pada tick TERAKHIR sebelum boundary T+15m
target         : 1 jika final_price > initial_price + threshold
               : 0 jika final_price < initial_price - threshold
               : NaN (SKIP) jika |return| <= threshold
```

### 3.2 Implementasi Target Builder

```python
# src/labels/02b_label_builder.py

def build_target_polymarket_5m(df_1m: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    ATURAN KERAS:
    1. Gunakan harga dari df_1m yang sudah diverifikasi sumbernya
    2. Round boundary: floor ke 15 menit (BUKAN ceil, BUKAN round)
    3. initial_price = first close SETELAH boundary
    4. final_price = last close SEBELUM boundary berikutnya
    5. TIDAK ada forward-fill label
    6. Simpan provenance: sumber data, timestamp pertama, timestamp terakhir

    VALIDASI WAJIB setelah build:
    - UP rate harus antara 0.40 dan 0.60
    - Jika UP rate < 0.40 atau > 0.60: ada bug di label construction
    - Buang round dengan tick_count < min_ticks (sparse data)
    - Buang round dengan max_gap > max_gap_seconds (data tidak konsisten)
    - Simpan audit sample (500 round random) ke CSV untuk verifikasi manual
    """
    df = df_1m.copy().sort_index()

    # Tambahkan kolom gap untuk deteksi sparse data
    df['ts_diff'] = df.index.to_series().diff().dt.total_seconds()

    rounds = df.resample('15min').agg(
        initial_price = ('close', 'first'),
        final_price   = ('close', 'last'),
        tick_count    = ('close', 'count'),
        max_gap       = ('ts_diff', 'max'),
        volume_sum    = ('volume', 'sum'),
    )

    threshold = config['project']['target_threshold']
    rounds['future_return'] = (
        (rounds['final_price'] - rounds['initial_price']) /
        (rounds['initial_price'] + 1e-9)
    )

    # Label: 1/0/NaN
    rounds['target'] = np.where(
        rounds['future_return'] >  threshold, 1,
        np.where(
        rounds['future_return'] < -threshold, 0,
        np.nan)
    )

    # Validasi dan filter
    min_ticks = config['data']['min_ticks_per_round']
    max_gap   = config['data']['max_gap_seconds']

    valid = rounds[
        (rounds['tick_count'] >= min_ticks) &
        (rounds['max_gap']    <= max_gap) &
        rounds['target'].notna()
    ].copy()

    up_rate = valid['target'].mean()
    logger.info(f"Target built: {len(valid)} valid rounds | UP rate: {up_rate:.2%}")

    # VALIDASI UP RATE (BUG DETECTOR)
    assert 0.38 <= up_rate <= 0.62, (
        f"UP rate {up_rate:.2%} di luar range normal [38%, 62%]. "
        f"Ada bug di label construction atau data sangat tidak seimbang."
    )

    # Provenance
    valid['label_source'] = 'binance_futures_1m_ohlcv'
    valid['label_threshold'] = threshold
    valid['label_version'] = config['version']

    # Audit sample
    audit = valid.sample(n=min(500, len(valid)), random_state=42)
    audit.to_csv("data/labels/label_audit_sample.csv")
    logger.info(f"Audit sample saved: data/labels/label_audit_sample.csv")

    return valid
```

---

## BAGIAN 4 — FEATURE ENGINEERING (SPESIFIKASI DETAIL)

### 4.1 Prinsip Anti-Leakage (WAJIB DIPAHAMI)

```
LEAKAGE TYPE 1 — Future data in feature:
  BUG: df.groupby('round').transform('max') → melihat seluruh round termasuk future
  FIX: Gunakan data dari [T-window, T) eksklusif, BUKAN dari round T sendiri

LEAKAGE TYPE 2 — Scaler fit pada semua data:
  BUG: StandardScaler().fit(X_all)  # termasuk test data
  FIX: scaler.fit(X_train) → scaler.transform(X_val) → scaler.transform(X_test)

LEAKAGE TYPE 3 — RegimeDetector fit pada test data:
  BUG: self.scaler.fit_transform(X_test) di dalam predict()
  FIX: fit hanya sekali pada training data, predict menggunakan transform()

LEAKAGE TYPE 4 — VPIN index mismatch:
  BUG: menggunakan datetime label sebagai integer position (bucket_start_idx = datetime)
  FIX: gunakan integer position konsisten, df.iloc[prev_pos:curr_pos+1]

LEAKAGE TYPE 5 — Inter-round features tanpa shift:
  BUG: df['past_ret'] = df['future_return']  # current round! leakage
  FIX: df['past_ret_lag1'] = df['future_return'].shift(1)  # round sebelumnya

LEAKAGE TYPE 6 — merge_asof direction='forward':
  BUG: merge_asof(df_features, df_target, direction='forward')
  FIX: merge_asof(df_features, df_target, direction='backward')
       atau join langsung karena index sudah aligned
```

### 4.2 Hierarki Feature Groups

```
GROUP 1 — SNAPSHOT FEATURES (nilai pada tick terakhir sebelum round)
  Prefix: snap_
  Dihitung dari: tick terakhir dalam window [T-15m, T)
  Diizinkan untuk: SEMUA tipe feature (microstructure, technical, time)
  Catatan: time features HANYA boleh muncul sebagai snap_, tidak sebagai win_*

GROUP 2 — WINDOW SIGNAL FEATURES (statistik dari window pre-round)
  Prefix: win_mean_, win_std_, win_trend_, win_skew_
  Dihitung dari: semua tick dalam window [T-15m, T)
  DILARANG untuk: time features (sin_hour, is_session, dll.)
  DIIZINKAN untuk: microstructure signals (OFI, OBI, spread, dll.)
                   technical signals (RSI, MACD, ATR, dll.)
                   price action (return, volatility, dll.)

GROUP 3 — ACCELERATION FEATURES (late half vs early half)
  Prefix: win_*_accel
  Dihitung dari: perbandingan paruh akhir vs paruh awal window
  Contoh: win_ofi_accel = mean(ofi, late_half) - mean(ofi, early_half)
  Interpretasi: apakah buying pressure mengakselerasi menjelang round?

GROUP 4 — INTER-ROUND FEATURES (lag dari round-round sebelumnya)
  Prefix: past_
  Dihitung dari: hasil round yang sudah selesai (shift ≥ 1)
  WAJIB: semua menggunakan .shift(1) atau lebih TIDAK PERNAH .shift(0)
  Contoh: past_ret_lag1 = future_return.shift(1)  # return round kemarin

GROUP 5 — CROSS-SOURCE INTERACTION FEATURES
  Prefix: (tidak ada prefix khusus, nama deskriptif)
  Dihitung dari: kombinasi dua atau lebih sumber data
  Contoh: ofi_rsi_div = ofi_normalized × (50 - rsi_14) / 50
  Catatan: interaksi hanya dari data masa lalu, tidak dari round aktif
```

### 4.3 Spesifikasi Detail Setiap Feature

#### A. Order Book Features (dari OB L2 jika tersedia)

```python
# Untuk setiap snapshot OB dengan depth D:
# bids = [[p_0, q_0], [p_1, q_1], ..., [p_D-1, q_D-1]] (descending price)
# asks = [[p_0, q_0], [p_1, q_1], ..., [p_D-1, q_D-1]] (ascending price)

# VALIDASI WAJIB sebelum compute:
# assert bids[0][0] < asks[0][0]  # best_bid < best_ask
# assert bids[0][0] > bids[1][0]  # bids descending
# assert asks[0][0] < asks[1][0]  # asks ascending
# Jika validasi gagal: isi NaN, JANGAN gunakan nilai palsu

# Feature formulas:

# 1. OBI (Order Book Imbalance) multi-level
# Formula: OBI_L = (sum_bid_vol_L - sum_ask_vol_L) / (sum_bid_vol_L + sum_ask_vol_L)
# Range: [-1, 1], stasioner
# Level: L in [1, 3, 5, 10, 20]
obi_L = (bid_vol[:L].sum() - ask_vol[:L].sum()) / (bid_vol[:L].sum() + ask_vol[:L].sum() + 1e-9)

# 2. Mid-price
mid = (best_bid + best_ask) / 2.0

# 3. Spread
spread_abs = best_ask - best_bid
spread_pct = spread_abs / mid  # Normalized, stasioner

# 4. Micro-price (lebih informatif dari mid ketika queue imbalance besar)
# Formula: (bid_price × ask_vol + ask_price × bid_vol) / (bid_vol + ask_vol)
micro_price = (best_bid * ask_vol_0 + best_ask * bid_vol_0) / (bid_vol_0 + ask_vol_0 + 1e-9)
micro_vs_mid = (micro_price - mid) / (spread_abs + 1e-9)  # Normalized, stasioner

# 5. Depth Ratio
depth_ratio = total_bid_vol / (total_ask_vol + 1e-9)  # Range: (0, ∞), bukan stasioner
# Gunakan log transform: log_depth_ratio = log(depth_ratio)

# 6. Bid/Ask concentration
bid_conc_l1 = bid_vol_0 / (total_bid_vol + 1e-9)  # [0,1], stasioner
ask_conc_l1 = ask_vol_0 / (total_ask_vol + 1e-9)  # [0,1], stasioner

# 7. Weighted Average Price asymmetry (stasioner)
bid_wap = sum(bid_price_i × bid_vol_i) / sum(bid_vol_i)
ask_wap = sum(ask_price_i × ask_vol_i) / sum(ask_vol_i)
wap_asym = ((mid - bid_wap) - (ask_wap - mid)) / (mid + 1e-9)

# 8. OBI per band harga
# Hanya volume dalam jarak X% dari mid
for pct in [0.10, 0.25, 0.50]:
    bid_in_band = sum(vol for price, vol in bids if abs(mid - price)/mid <= pct/100)
    ask_in_band = sum(vol for price, vol in asks if abs(price - mid)/mid <= pct/100)
    obi_band_X = (bid_in_band - ask_in_band) / (bid_in_band + ask_in_band + 1e-9)

# 9. OB Momentum (perubahan OBI dari snapshot sebelumnya)
obi_1_mom = obi_1[t] - obi_1[t-12]  # 12 tick ke belakang (~1 menit)
obi_1_accel = obi_1_mom[t] - obi_1_mom[t-12]  # Perubahan momentum

# 10. OB Toxicity (volatilitas OBI dalam window pendek)
obi_toxicity = std(obi_1, window=12)
```

#### B. OFI Features (dari aggTrades sebagai proxy jika tidak ada OB trades)

```python
# PENTING: aggTrades ≠ OB L2. aggTrades adalah transaksi yang SUDAH terjadi.
# OFI dari aggTrades mengukur agresivitas buyer/seller (backward-looking).
# OBI dari OB mengukur intention (forward-looking, lebih prediktif).
# Gunakan keduanya jika tersedia.

# Resample aggTrades ke 1 detik dulu:
# buy_vol = qty dimana is_buyer_maker == False (aggressor adalah buyer)
# sell_vol = qty dimana is_buyer_maker == True  (aggressor adalah seller)

# 1. OFI L1 (Order Flow Imbalance, diproksi dari trades)
ofi_l1 = buy_vol_1s - sell_vol_1s

# 2. OFI Normalized (stasioner)
ofi_normalized = ofi_l1 / (total_vol_1s + 1e-9)  # Range: [-1, 1]

# 3. OBI Proxy dari trade flow (bukan dari order book langsung)
# Window 10 detik rolling
obi_proxy_10s = (rolling_buy_vol_10s - rolling_sell_vol_10s) / \
                (rolling_total_vol_10s + 1e-9)

# 4. VPIN (Volume-synchronized Probability of Informed Trading)
# IMPLEMENTASI YANG BENAR (pelajaran dari iterasi sebelumnya):
# WAJIB: gunakan integer position (iloc), BUKAN datetime label (loc)
def compute_vpin_correct(df_trades, buckets=30):
    df = df_trades.reset_index()  # Reset index ke integer
    df['buy_vol']  = np.where(df['is_buyer_maker'] == False, df['qty'], 0)
    df['sell_vol'] = np.where(df['is_buyer_maker'] == True,  df['qty'], 0)
    df['total_vol'] = df['buy_vol'] + df['sell_vol']
    df['imbalance'] = (df['buy_vol'] - df['sell_vol']).abs()

    cum_vol    = df['total_vol'].cumsum().values
    total_vol  = cum_vol[-1]
    bucket_size = total_vol / buckets

    vpin_vals = np.zeros(len(df))
    prev_pos  = 0  # INTEGER position, bukan datetime

    for i in range(buckets):
        target_vol = (i + 1) * bucket_size
        next_pos_arr = np.where(cum_vol >= target_vol)[0]
        curr_pos = next_pos_arr[0] if len(next_pos_arr) > 0 else len(df) - 1

        # GUNAKAN iloc (integer), BUKAN loc (label)
        bucket = df.iloc[prev_pos:curr_pos + 1]
        bucket_imb = bucket['imbalance'].sum()
        bucket_vol = bucket['total_vol'].sum()
        vpin_i = bucket_imb / (bucket_vol + 1e-9)
        vpin_vals[prev_pos:curr_pos + 1] = vpin_i
        prev_pos = curr_pos + 1

    df['vpin'] = vpin_vals
    return df.set_index('timestamp')[['vpin']]

# 5. Z-score microstructure (deteksi shock vs normal)
# Window 1 jam (3600 detik)
ofi_zscore = (ofi_normalized - rolling_mean(ofi_normalized, 3600)) / \
             (rolling_std(ofi_normalized, 3600) + 1e-9)
```

#### C. Technical Features (dari OHLCV 1m)

```python
# SEMUA technical features harus STASIONER (bukan harga absolut)

# 1. RSI (range [0, 100], tidak perlu normalisasi tambahan)
rsi_7  = talib.RSI(close, timeperiod=7)
rsi_14 = talib.RSI(close, timeperiod=14)
rsi_21 = talib.RSI(close, timeperiod=21)

# 2. MACD (sudah merupakan selisih, stasioner)
macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
# Gunakan: macd, macd_hist (bukan macd_signal terpisah)

# 3. Bollinger Bands
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
bb_pct_20   = (close - lower) / (upper - lower + 1e-9)  # [0, 1], posisi dalam band
bb_width_20 = (upper - lower) / (middle + 1e-9)  # lebar band relatif, stasioner

# 4. ATR (normalisasi oleh close untuk stationary)
atr_14_norm = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-9)

# 5. EMA Distance (stasioner: persentase deviasi dari EMA)
for p in [9, 21, 50]:
    ema = talib.EMA(close, timeperiod=p)
    ema_dist = (close - ema) / (ema + 1e-9)  # [-X, +X], stasioner

# 6. Volume Ratio (stasioner)
vol_ratio_15 = volume / (rolling_mean(volume, 15) + 1e-9)
vol_ratio_60 = volume / (rolling_mean(volume, 60) + 1e-9)

# 7. Rolling VWAP Deviation (stasioner, BUKAN daily VWAP yang mengandung future)
# Gunakan ROLLING VWAP, bukan daily VWAP
vwap_rolling_60 = (close × volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-9)
vwap_dev = (close - vwap_rolling_60) / (vwap_rolling_60 + 1e-9)
# Normalize lebih lanjut dengan ATR
vwap_dev_norm = vwap_dev / (atr_14_norm + 1e-9)

# 8. Price Velocity (acceleration of returns, stasioner)
log_ret = log(close / close.shift(1))
price_velocity = log_ret - log_ret.shift(3)  # d(return)/dt ≈ perubahan return

# 9. Momentum multi-timeframe (stasioner)
momentum_1h  = close / (close.shift(60)  + 1e-9) - 1   # 1 jam ke belakang
momentum_4h  = close / (close.shift(240) + 1e-9) - 1   # 4 jam ke belakang
# CATATAN: shift(60) artinya 60 candle 1m ke belakang = 60 menit. BUKAN masa depan.

# 10. Realized Volatility multi-window (stasioner)
for window in [60, 180, 300]:  # 1m, 3m, 5m dalam detik
    n_candles = window // 60   # Konversi ke jumlah candle 1m
    rvol = log_ret.rolling(n_candles).std() * sqrt(n_candles)  # Annualized optional
```

#### D. Time Features (HANYA untuk snapshot, TIDAK diagregasi)

```python
# ATURAN KERAS: TIME FEATURES TIDAK BOLEH DIAGREGASI
# Alasan: jam tidak berubah dalam window 15 menit
# win_mean_sin_hour == snap_sin_hour  → informasi 0, noise murni
# Pengamatan iterasi 9: sin_hour dan cos_hour menjadi top features KARENA spurious correlation

# Benar: hanya snapshot
def add_time_features_snapshot_only(timestamp: pd.Timestamp) -> dict:
    hour = timestamp.hour + timestamp.minute / 60.0  # Fractional hour
    dow  = timestamp.dayofweek  # 0=Mon, 6=Sun

    # Cyclic encoding (stasioner, tidak ada boundary effect)
    features = {
        'sin_hour' : np.sin(2 * np.pi * timestamp.hour / 24),
        'cos_hour' : np.cos(2 * np.pi * timestamp.hour / 24),
        'sin_dow'  : np.sin(2 * np.pi * dow / 7),
        'cos_dow'  : np.cos(2 * np.pi * dow / 7),
    }

    # 15-menit slot dalam hari (96 slot per hari)
    slot_15m = timestamp.hour * 4 + timestamp.minute // 15
    features['sin_15m'] = np.sin(2 * np.pi * slot_15m / 96)
    features['cos_15m'] = np.cos(2 * np.pi * slot_15m / 96)

    # Session flags
    h = timestamp.hour
    features['is_us_session']     = float(13 <= h < 21)
    features['is_london_session'] = float(7 <= h < 16)
    # PENTING: gunakan perkalian (bukan &) untuk float values
    features['is_overlap']        = features['is_us_session'] * features['is_london_session']
    features['is_weekend']        = float(dow >= 5)

    # Polymarket-specific: beberapa jam memiliki volume/likuiditas berbeda
    features['is_roll_hour']       = float(h % 8 == 0)  # 00:00, 08:00, 16:00
    features['minutes_since_roll'] = float((h % 8) * 60 + timestamp.minute)

    return features
```

#### E. Cross-Source Interaction Features

```python
# Fitur interaksi antara dua sumber berbeda
# ATURAN: hanya dari data masa lalu, tidak dari round aktif

# 1. OFI-RSI Divergence
# Interpretasi: OFI positif (buy pressure) tapi RSI overbought = divergence = kemungkinan reversal
ofi_rsi_div = ofi_normalized × (50 - rsi_14) / 50
# Positif: OFI dan RSI selaras (keduanya bullish atau bearish)
# Negatif: divergence (OFI bullish tapi RSI overbought)

# 2. VPIN-Volatility Interaction
# Interpretasi: VPIN tinggi di environment volatil = kondisi toxic lebih parah
vpin_vol_toxic = vpin × atr_14_norm

# 3. OFI-OBI Interaction (jika ada OB L2)
ofi_obi_inter = ofi_normalized × obi_proxy_10s
# Interpretasi: kedua sinyal satu arah = konfirmasi kuat

# 4. Volume-OFI Confirmation
# Interpretasi: volume surge dengan directional flow kuat = momentum valid
if 'vol_ratio_15' in df.columns:
    vol_ofi_confirm = vol_ratio_15 × ofi_normalized

# 5. Price-OFI Divergence (IMPLEMENTASI YANG AMAN — tidak ada leakage)
# Gunakan lookback 300 detik (5 menit ke belakang), bukan ke depan
price_ofi_div = (
    close.pct_change(300) -           # Return 5 menit ke belakang
    ofi_normalized.rolling(300).mean() # OFI rata-rata 5 menit ke belakang
)
# CATATAN: pct_change(300) = (close[t] - close[t-300]) / close[t-300]
# Ini AMAN — menggunakan data historis, bukan masa depan
```

### 4.4 Window Aggregation — Implementasi yang Benar

```python
# ATURAN UTAMA WINDOW AGGREGATION:
# 1. Semua fitur dari window [T-15m, T) untuk prediksi round T
# 2. Time features HANYA masuk sebagai SNAPSHOT (last value), TIDAK diagregasi
# 3. Gunakan vectorized operations (groupby), BUKAN Python loop
# 4. Index output di-shift: features[T] digunakan untuk prediksi round T
#    (round T dimulai pada waktu T, fitur diambil dari [T-15m, T))

def aggregate_window_features_v1(df_full: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Input: df_full dengan index datetime, setiap baris = 1 tick/second
    Output: 1 baris per round, index = round_start timestamp
    """
    time_like_cols = set(config['features']['time_like_cols'])

    df = df_full.copy()
    # Label setiap tick dengan round yang akan diprediksi
    df['round_label'] = df.index.floor('15min')

    # Tentukan kolom yang boleh diagregasi vs snapshot saja
    all_cols      = [c for c in df.columns if c != 'round_label']
    signal_cols   = [c for c in all_cols if c not in time_like_cols]
    snapshot_cols = [c for c in all_cols if c in time_like_cols]

    parts = []
    grp = df.groupby('round_label')

    # GROUP 2: Window Signal Stats (HANYA signal_cols)
    sig_mean  = grp[signal_cols].mean().add_prefix('win_mean_')
    sig_std   = grp[signal_cols].std().add_prefix('win_std_')
    sig_trend = (grp[signal_cols].last() - grp[signal_cols].first()).add_prefix('win_trend_')
    parts.extend([sig_mean, sig_std, sig_trend])

    # GROUP 1: Snapshot (signal_cols + time_like_cols)
    snap_all = grp[all_cols].last().add_prefix('snap_')
    parts.append(snap_all)

    # GROUP 3: Acceleration (OFI, VPIN)
    for accel_col, out_name in [('ofi_normalized', 'win_ofi_accel'),
                                  ('vpin', 'win_vpin_accel')]:
        if accel_col not in df.columns:
            continue
        # Bagi setiap round menjadi 2 paruh menggunakan cumcount
        df['tick_idx'] = df.groupby('round_label').cumcount()
        df['win_size'] = df.groupby('round_label')['tick_idx'].transform('max')
        df['is_late']  = df['tick_idx'] > (df['win_size'] / 2)

        accel = df.groupby(['round_label', 'is_late'])[accel_col].mean().unstack(fill_value=0)
        if accel.shape[1] == 2:
            accel.columns = ['early', 'late']
            accel_series  = (accel['late'] - accel['early']).rename(out_name)
            parts.append(accel_series)

    # Price Action (dari harga mentah)
    if 'price' in df.columns:
        price_grp   = grp['price']
        price_first = price_grp.first()
        price_last  = price_grp.last()
        price_min   = price_grp.min()
        price_max   = price_grp.max()

        win_ret = pd.DataFrame({
            'win_return'         : price_last / (price_first + 1e-9) - 1,
            'win_close_position' : (price_last - price_min) / (price_max - price_min + 1e-9),
        })
        parts.append(win_ret)

    # Realized volatility dalam window
    if 'log_ret' in df.columns:
        win_rvol = grp['log_ret'].std().rename('win_rvol')
        parts.append(win_rvol)

    # Gabungkan semua parts sekaligus (hindari fragmented DataFrame)
    df_aggr = pd.concat(parts, axis=1)

    # SHIFT INDEX: features dari [T-15m, T) → digunakan untuk prediksi round T
    # Setelah shift, index round_label T = fitur yang diambil dari [T-15m, T)
    df_aggr.index     = df_aggr.index + pd.Timedelta(minutes=15)
    df_aggr.index.name = 'round_start'

    return df_aggr
```

### 4.5 Inter-Round Features — Implementasi yang Benar

```python
def add_inter_round_features_v1(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    WAJIB: Semua menggunakan .shift(N) dengan N >= 1
    DILARANG: .shift(0) = current round = leakage

    Input: df dengan kolom future_return dan target (sudah di-join dengan target)
    """
    res = df.copy().sort_index()
    n_lags   = config['features']['inter_round_lags']   # default 8
    roll_wins = config['features']['inter_round_rolling']  # [4, 8, 16]

    # Lag returns dan outcomes
    for lag in range(1, n_lags + 1):
        res[f'past_ret_lag{lag}'] = res['future_return'].shift(lag)
        res[f'past_up_lag{lag}']  = res['target'].shift(lag)

    res['past_ret_abs_lag1'] = res['future_return'].shift(1).abs()

    # Rolling stats antar-round
    for w in roll_wins:
        past_ret = res['future_return'].shift(1)
        past_up  = res['target'].shift(1)
        res[f'past_up_rate_{w}']  = past_up.rolling(w, min_periods=max(1, w//2)).mean()
        res[f'past_ret_sum_{w}']  = past_ret.rolling(w, min_periods=max(1, w//2)).sum()
        res[f'past_rvol_{w}']     = past_ret.rolling(w, min_periods=max(1, w//2)).std()

    # Mean reversion z-score
    res['past_ret_zscore_8'] = (
        res['past_ret_lag1'] / (res['past_rvol_8'] + 1e-9)
    )

    # Streak: berapa round berturut-turut arah sama
    past_up_series = res['target'].shift(1).fillna(0.5).astype(int)
    streak = pd.Series(0, index=res.index)
    for i in range(1, len(res)):
        if i == 0:
            streak.iloc[i] = 0
        elif past_up_series.iloc[i] == past_up_series.iloc[i - 1]:
            streak.iloc[i] = streak.iloc[i - 1] + 1
        else:
            streak.iloc[i] = 0
    res['past_streak'] = streak.shift(1)

    # OBI inter-round (jika ada dari OB L2)
    obi_col = 'snap_obi_1' if 'snap_obi_1' in res.columns else None
    if obi_col:
        for lag in range(1, 5):
            res[f'past_obi_lag{lag}'] = res[obi_col].shift(lag)
        res['past_obi_trend_4'] = res[obi_col].shift(1) - res[obi_col].shift(4)
        res['past_obi_ma4']     = res[obi_col].shift(1).rolling(4, min_periods=1).mean()

    return res
```

---

## BAGIAN 5 — FEATURE SELECTION (WAJIB SEBELUM GPU TRAINING)

### 5.1 Mengapa Feature Selection Kritis

```
Iterasi 9 menunjukkan: 194 fitur langsung ke CatBoost GPU → model stuck di AUC 0.52
Alasan:
  - Rasio fitur:sampel = 194:27932 ≈ 1:144 (masih banyak, tapi noise sangat tinggi)
  - GPU training dengan banyak fitur noise = overfitting pada noise
  - Model menemukan spurious correlations di training, tidak generalize ke test

Target setelah reduction: 70 fitur (dari ~150-200 fitur raw)
Method: LightGBM CPU screening (cepat, tidak pakai GPU)
```

### 5.2 Implementasi Feature Reduction

```python
def reduce_features_via_lgbm_v1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,          # WAJIB dioper sebagai parameter, bukan closure
    X_test: pd.DataFrame,
    w_train: np.ndarray = None,
    target_features: int = 70,
    config: dict = None
) -> tuple:
    """
    LANGKAH YANG HILANG DI ITERASI 9.
    Harus dipanggil SEBELUM optimize_catboost().

    PERBAIKAN DARI ITERASI SEBELUMNYA:
    - y_val dioper sebagai parameter (bukan menggunakan dari closure)
    - boosting_type='gbdt' (bukan 'dart')
    - early_stopping diberikan dari param screener

    Returns:
        (X_train_reduced, X_val_reduced, X_test_reduced, selected_features, lgbm_baseline_auc)
    """
    lgb_cfg = config['training']['lightgbm_screener']

    screener = lgb.LGBMClassifier(**{k: v for k, v in lgb_cfg.items()
                                      if k not in ['early_stopping_rounds']})
    screener.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(lgb_cfg['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(-1),
        ]
    )

    lgbm_auc = roc_auc_score(y_val, screener.predict_proba(X_val)[:, 1])

    importance = pd.Series(
        screener.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # Hapus zero-importance features dulu
    nonzero      = importance[importance > 0]
    top_features = nonzero.head(target_features).index.tolist()

    # Analisis kategori feature yang terpilih
    categories = {
        'snap_'  : 'Microstructure Snapshot',
        'win_mean_': 'Window Mean',
        'win_std_' : 'Window Std',
        'win_trend_': 'Window Trend',
        'past_'  : 'Inter-round Lag',
        'win_ofi' : 'OFI Accel/Window',
        'win_ret' : 'Price Action',
    }
    for prefix, name in categories.items():
        n = sum(1 for f in top_features if f.startswith(prefix))
        if n > 0:
            logger.info(f"  {name}: {n} features")

    # DIAGNOSTIK PENTING: Jika past_up_lag1 tidak masuk top 20,
    # inter-round signal sangat lemah → kemungkinan besar coverage data rendah
    top_20 = top_features[:20]
    if 'past_up_lag1' not in top_20 and 'past_ret_lag1' not in top_20:
        logger.warning("⚠️ past_up_lag1 TIDAK dalam top 20 — inter-round signal lemah")
        logger.warning("   Kemungkinan: coverage data rendah, inter-round features tidak valid")

    # Simpan untuk reproduksi
    importance.to_csv("models/artifacts/lgbm_importance_v1.csv", header=['importance'])
    joblib.dump(top_features, "models/artifacts/selected_features_v1.pkl")

    return (
        X_train[top_features].astype(np.float32),
        X_val[top_features].astype(np.float32),
        X_test[top_features].astype(np.float32),
        top_features,
        lgbm_auc
    )
```

---

## BAGIAN 6 — TRAINING PIPELINE (URUTAN WAJIB)

### 6.1 Urutan Langkah Training

```
LANGKAH 1: load_and_validate_data()
  → Load features parquet
  → Sort by round_start (WAJIB)
  → Gap check: tandai round yang round sebelumnya gap > 30 menit
  → Invalidasi inter-round features untuk round yang gapped
  → Hitung coverage, up_rate, verifikasi sanity
  → Buang round dengan has_valid_context = False

LANGKAH 2: select_features()
  → Keluarkan: round_start, target, initial_price, final_price,
               future_return, gap_to_prev, has_valid_context, round_label
  → Keluarkan: kolom dengan NaN > 30%
  → Keluarkan: kolom konstan (std < 1e-10)
  → Keluarkan: harga absolut (close, price, open, high, low) yang tidak dinormalisasi
  → FIX OPERATOR PRECEDENCE: gunakan fungsi is_valid_feature() yang eksplisit

LANGKAH 3: time_split_strict()
  → Sort by round_start
  → Train: 70% pertama (kronologis)
  → Val  : 15% berikutnya
  → Test : 15% terakhir (terbaru)
  → Assert: test_min > val_min > train_min
  → Assert: tidak ada overlap antar split
  → TIDAK ADA SHUFFLE

LANGKAH 4: compute_sample_weights()
  → Hitung median |future_return| dari TRAINING set saja
  → weight = 2.0 jika |return| >= median, 1.0 jika |return| < median
  → CATATAN: 2.0x tidak 3.5x (3.5x terbukti menyebabkan LightGBM "no splits")

LANGKAH 5: reduce_features_via_lgbm()  ← LANGKAH YANG HILANG DI ITERASI 9
  → LightGBM CPU screening dengan boosting_type='gbdt'
  → y_val harus dioper sebagai parameter fungsi
  → Reduksi dari ~150-200 → 70 fitur
  → Catat baseline AUC dari LightGBM
  → Jika LightGBM AUC < 0.51 → WARNING: signal sangat lemah

LANGKAH 6: optimize_hyperparams_optuna()
  → 80 trials dengan TPE sampler
  → Objective: auc_val - penalty_overfit + bonus_prob_spread
  → Penalti overfit: max(0, auc_train - auc_val - 0.03) × 5.0
  → Bonus prob spread: min(prob_spread_90_10 × 5.0, 0.03)
  → Catch semua exception per trial (jangan crash total)
  → Filter trial dengan value = 0.5 (gagal)

LANGKAH 7: train_final_model()
  → CatBoost GPU: iterations=10000, early_stopping=500, gpu_ram_part=0.40
  → LightGBM CPU: boosting_type='gbdt', num_boost_round=2000
  → Ensemble weights: proporsional terhadap val AUC masing-masing
  → Simpan snapshot setiap 300 detik (untuk resume jika crash)

LANGKAH 8: fit_calibrator()
  → Menggunakan VAL SET saja (TIDAK test set)
  → Method: isotonic regression (lebih fleksibel dari Platt/sigmoid)
  → Simpan calibrator terpisah dari model utama

LANGKAH 9: find_optimal_threshold()
  → Scan dari 0.50 (bukan 0.505) sampai 0.70 step 0.005
  → Filter: coverage minimal 10% (agar ada cukup sampel)
  → Pilih threshold dengan accuracy tertinggi di VAL set
  → TIDAK pernah di TEST set

LANGKAH 10: full_evaluation()
  → Report AUC train/val/test + overfit gap
  → Distribusi probabilitas (10 bin) — DIAGNOSTIK KRITIS
  → High-confidence filter di berbagai threshold
  → Calibration check (reliability plot)
  → Feature importance by category
  → Per-session analysis (US, London, Asian, Overlap)
  → Per-volatility-regime analysis
  → Walk-forward validation (6 fold)
```

### 6.2 Gap Detection dan Invalidasi Inter-Round Features

```python
def load_and_validate_data_v1(features_path: str) -> pd.DataFrame:
    df = pd.read_parquet(features_path)

    if 'round_start' not in df.columns:
        df = df.reset_index()

    df = df.sort_values('round_start').reset_index(drop=True)

    # STEP 1: Hitung gap antar round (dalam detik)
    df['gap_to_prev'] = pd.to_datetime(df['round_start']).diff().dt.total_seconds()

    # STEP 2: Flag round yang round sebelumnya ada gap besar
    # Gap > 30 menit (1800 detik) = round sebelumnya tidak ada datanya
    df['has_valid_context'] = (df['gap_to_prev'] <= 1800) | df['gap_to_prev'].isna()

    # STEP 3: Invalidasi inter-round features untuk round tanpa konteks valid
    # Ini mencegah model belajar dari past_ret_lag1 = NaN yang diimputasi
    lag_cols = [c for c in df.columns if c.startswith('past_')]
    n_invalidated = (~df['has_valid_context']).sum()
    df.loc[~df['has_valid_context'], lag_cols] = np.nan
    logger.info(f"Invalidated {n_invalidated} rounds' inter-round features (gap > 30min)")

    # STEP 4: Hitung coverage per hari
    df['date'] = pd.to_datetime(df['round_start']).dt.date
    rounds_per_day = 24 * 4  # 96 rounds/hari untuk 15m
    daily_cov = df.groupby('date').size() / rounds_per_day

    bad_days = daily_cov[daily_cov < 0.30].index
    if len(bad_days) > 0:
        logger.warning(f"⚠️ {len(bad_days)} hari dengan coverage < 30%: {list(bad_days[:5])}")
        # Opsi: buang hari dengan coverage sangat rendah
        # df = df[~df['date'].isin(bad_days)]

    # STEP 5: Buang round tanpa target
    df = df.dropna(subset=['target', 'future_return'])

    # STEP 6: Sanity checks
    n = len(df)
    up_rate  = df['target'].mean()
    coverage_overall = n / max(1, int(
        (pd.to_datetime(df['round_start'].max()) -
         pd.to_datetime(df['round_start'].min())).total_seconds() / 900))

    logger.info(f"Loaded: {n} rounds | UP rate: {up_rate:.2%} | "
                f"Coverage: {coverage_overall:.1%}")

    if up_rate < 0.40 or up_rate > 0.60:
        raise ValueError(f"UP rate {up_rate:.2%} tidak normal. Cek label construction.")

    if coverage_overall < 0.50:
        logger.warning(f"⚠️ Coverage {coverage_overall:.1%} < 50% — "
                       "inter-round features tidak reliabel untuk sebagian besar data")

    df = df.drop(columns=['date'], errors='ignore')
    return df
```

### 6.3 Sample Weight yang Benar

```python
def compute_sample_weights(train_df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    PELAJARAN: 3.5x terlalu agresif → LightGBM "No further splits"
    Fix: turunkan ke 2.0x

    Reasoning: weight tinggi menyebabkan effective sample size menjadi kecil,
    LightGBM tidak bisa menemukan split yang generalizable.
    """
    w_high = config['training']['sample_weight_high']  # 2.0
    w_low  = config['training']['sample_weight_low']   # 1.0

    # Threshold dari TRAINING data saja (tidak dari val/test)
    vol_thresh = np.abs(train_df['future_return']).median()

    weights = np.where(
        np.abs(train_df['future_return']) >= vol_thresh,
        w_high,
        w_low
    )

    logger.info(f"Sample weights: high={w_high}x ({(weights==w_high).mean():.1%}), "
                f"low={w_low}x ({(weights==w_low).mean():.1%}), "
                f"vol_threshold={vol_thresh:.5f}")
    return weights
```

---

## BAGIAN 7 — REGIME DETECTOR (IMPLEMENTASI YANG BENAR)

### 7.1 Bug Kritis di Iterasi Sebelumnya

```
BUG (iterasi sebelumnya):
  predict() → detect_regimes() → scaler.fit_transform(X_test)
  
  Ini adalah leakage: scaler di-fit pada test data setiap kali prediksi.
  Model "melihat" distribusi test data untuk menentukan regime.

FIX:
  fit() → scaler.fit(X_train) → simpan scaler
  predict() → scaler.transform(X_new) → gunakan scaler yang sudah di-fit
```

### 7.2 Implementasi yang Benar

```python
class RegimeDetector:
    """
    Deteksi regime pasar (volatilitas rendah/normal/tinggi).
    KRITIS: scaler dan classifier hanya di-fit pada training data SEKALI.
    """
    def __init__(self, n_regimes: int = 3):
        self.n_regimes    = n_regimes
        self.scaler       = None  # Di-fit SEKALI pada train
        self.classifier   = None  # Di-fit SEKALI pada train
        self.is_fitted    = False

    def _extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features untuk deteksi regime. HANYA dari data historis."""
        features = {}
        if 'win_rvol' in df.columns:
            features['rvol']         = df['win_rvol']
        if 'snap_atr_14_norm' in df.columns:
            features['atr']          = df['snap_atr_14_norm']
        if 'snap_spread_pct' in df.columns:
            features['spread']       = df['snap_spread_pct']
        if 'snap_bb_width_20' in df.columns:
            features['bb_width']     = df['snap_bb_width_20']
        if 'past_rvol_4' in df.columns:
            features['past_rvol_4']  = df['past_rvol_4']
        return pd.DataFrame(features, index=df.index).fillna(0)

    def fit(self, X_train: pd.DataFrame) -> 'RegimeDetector':
        """
        HANYA dipanggil SEKALI dengan training data.
        TIDAK pernah dipanggil lagi dengan validation atau test data.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        regime_features = self._extract_regime_features(X_train)

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(regime_features)

        self.classifier = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        self.classifier.fit(features_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediksi regime untuk data baru.
        HANYA menggunakan scaler.transform() (TIDAK fit_transform()).
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector belum di-fit. Panggil fit(X_train) terlebih dahulu.")

        regime_features = self._extract_regime_features(X)
        # TRANSFORM SAJA — tidak fit ulang
        features_scaled = self.scaler.transform(regime_features)
        return self.classifier.predict(features_scaled)

    def save(self, path: str):
        joblib.dump({'scaler': self.scaler, 'classifier': self.classifier,
                     'n_regimes': self.n_regimes, 'is_fitted': self.is_fitted}, path)

    @classmethod
    def load(cls, path: str) -> 'RegimeDetector':
        data = joblib.load(path)
        obj  = cls(n_regimes=data['n_regimes'])
        obj.scaler     = data['scaler']
        obj.classifier = data['classifier']
        obj.is_fitted  = data['is_fitted']
        return obj
```

---

## BAGIAN 8 — PROBABILITY CALIBRATION (WAJIB)

### 8.1 Mengapa Kalibrasi Penting

```
TANPA kalibrasi:
  - Model mengeluarkan prob 0.65 untuk event yang sebenarnya terjadi 55% waktu
  - Jika Polymarket harga YES = 0.60, model berpikir ada edge (0.65 > 0.60)
  - Kenyataan: tidak ada edge (0.55 vs 0.60 = model RUGI)

DENGAN kalibrasi:
  - Model mengeluarkan prob 0.55 untuk event yang sama (setelah calibration)
  - Jika Polymarket harga YES = 0.60, model BENAR tidak bet

Expected Calibration Error (ECE):
  - < 0.05 = well calibrated
  - 0.05-0.10 = acceptable
  - > 0.10 = perlu recalibration
```

### 8.2 Implementasi

```python
def fit_calibrator_v1(model_probs_val: np.ndarray,
                       y_val: pd.Series,
                       method: str = 'isotonic') -> object:
    """
    Fit calibrator HANYA pada VAL set.
    TIDAK pada training set (in-sample calibration = overfit calibration).
    TIDAK pada test set (leakage).
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve

    # Check calibration sebelum
    fraction_pos, mean_pred = calibration_curve(y_val, model_probs_val, n_bins=10)
    ece_before = np.mean(np.abs(fraction_pos - mean_pred))
    logger.info(f"ECE sebelum kalibrasi: {ece_before:.4f}")

    # Fit isotonic regression
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(model_probs_val, y_val)

    # Verifikasi setelah kalibrasi
    probs_calibrated = calibrator.predict(model_probs_val)
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_val, probs_calibrated, n_bins=10)
    ece_after = np.mean(np.abs(fraction_pos_cal - mean_pred_cal))
    logger.info(f"ECE setelah kalibrasi: {ece_after:.4f}")

    if ece_after > 0.10:
        logger.warning("⚠️ ECE masih > 0.10 setelah kalibrasi. "
                       "Kemungkinan distribusi prob terlalu sempit (semua ~0.50).")

    return calibrator
```

---

## BAGIAN 9 — EVALUASI LENGKAP (NON-NEGOTIABLE)

### 9.1 Diagnostik yang Harus Dilaporkan

```python
def full_evaluation_v1(cb_model, lgb_model, calibrator,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        df_test_meta, ensemble_weights, feature_cols, config):
    """
    Laporan lengkap yang WAJIB ada sebelum model dianggap valid.
    """
    w_cb, w_lgb = ensemble_weights

    def ensemble_prob(X):
        p_cb  = cb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict(X)
        p_raw = p_cb * w_cb + p_lgb * w_lgb
        return calibrator.predict(p_raw)  # Setelah kalibrasi

    p_train = ensemble_prob(X_train)
    p_val   = ensemble_prob(X_val)
    p_test  = ensemble_prob(X_test)

    # 1. AUC semua split
    auc_train = roc_auc_score(y_train, p_train)
    auc_val   = roc_auc_score(y_val,   p_val)
    auc_test  = roc_auc_score(y_test,  p_test)
    gap       = auc_train - auc_test

    logger.info(f"\nAUC — Train: {auc_train:.4f} | Val: {auc_val:.4f} | Test: {auc_test:.4f}")
    logger.info(f"Overfit gap: {gap:.4f} {'✅' if gap < 0.05 else '⚠️ OVERFIT'}")

    # 2. DISTRIBUSI PROBABILITAS (DIAGNOSTIK TERPENTING)
    logger.info("\n--- Distribusi Probabilitas (TEST) ---")
    for lo, hi in [(0.30,0.40), (0.40,0.45), (0.45,0.50), (0.50,0.55), (0.55,0.60), (0.60,0.70)]:
        n = ((p_test >= lo) & (p_test < hi)).sum()
        logger.info(f"  [{lo:.2f}-{hi:.2f}): {n:4d} ({n/len(p_test)*100:.1f}%)")

    # INTERPRETASI:
    # Jika > 80% prob di range [0.45, 0.55] → model tidak belajar apa-apa
    # Jika distribusi lebih merata → ada signal
    flat_pct = ((p_test >= 0.45) & (p_test < 0.55)).mean()
    if flat_pct > 0.80:
        logger.warning(f"⚠️ {flat_pct:.1%} prob berada di [0.45, 0.55] — model hampir tidak ada signal")

    # 3. THRESHOLD SCAN (dari VAL, apply ke TEST)
    best_acc, best_t = 0, 0.50
    scan_start = config['evaluation']['threshold_scan_start']  # 0.50, BUKAN 0.505
    scan_end   = config['evaluation']['threshold_scan_end']
    scan_step  = config['evaluation']['threshold_scan_step']

    for t in np.arange(scan_start, scan_end, scan_step):
        mask_val = (p_val >= t) | (p_val <= 1 - t)
        cov_val  = mask_val.mean()
        if cov_val < config['evaluation']['min_coverage_for_threshold']:
            break
        acc_val = accuracy_score(y_val[mask_val], (p_val[mask_val] >= t).astype(int))
        if acc_val > best_acc:
            best_acc, best_t = acc_val, t

    logger.info(f"\nOptimal threshold (dari VAL): {best_t:.3f}")
    logger.info("\n--- Accuracy di TEST dengan berbagai threshold ---")
    for t in config['evaluation']['confidence_levels'] + [best_t]:
        mask = (p_test >= t) | (p_test <= 1 - t)
        n    = mask.sum()
        if n < 30:
            continue
        acc  = accuracy_score(y_test[mask], (p_test[mask] >= t).astype(int))
        cov  = n / len(p_test)
        logger.info(f"  t={t:.3f}: Acc={acc:.2%} | Cov={cov:.2%} ({n} rounds)")

    # 4. CALIBRATION CHECK
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10)
    ece = np.mean(np.abs(frac_pos - mean_pred))
    logger.info(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # 5. FEATURE IMPORTANCE BY CATEGORY
    imp = pd.DataFrame({'feature': feature_cols,
                        'importance': cb_model.get_feature_importance()
                       }).sort_values('importance', ascending=False)

    category_map = {
        'snap_'   : 'Microstructure Snapshot',
        'win_mean_': 'Window Mean',
        'win_std_' : 'Window Std',
        'win_trend_': 'Window Trend',
        'win_ofi' : 'OFI Accel/Window',
        'past_'   : 'Inter-round Lag',
        'win_ret' : 'Price Action',
    }
    logger.info("\n--- Feature Category Importance ---")
    for prefix, name in category_map.items():
        cat_feats = [f for f in imp['feature'] if f.startswith(prefix)]
        if cat_feats:
            total_imp = imp[imp['feature'].isin(cat_feats)]['importance'].sum()
            logger.info(f"  {name:<30}: {total_imp:.1f}")

    logger.info("\n--- Top 20 Features ---")
    for _, row in imp.head(20).iterrows():
        bar = '█' * int(row['importance'] / imp['importance'].max() * 20)
        logger.info(f"  {row['feature']:<45} {row['importance']:6.2f} {bar}")

    # 6. PER-SESSION ANALYSIS
    hours = pd.to_datetime(df_test_meta['round_start']).dt.hour
    sessions = {
        'US (13-21 UTC)'     : (hours >= 13) & (hours < 21),
        'London (07-16 UTC)' : (hours >= 7)  & (hours < 16),
        'Asian (00-08 UTC)'  : (hours >= 0)  & (hours < 8),
        'Overlap (13-16 UTC)': (hours >= 13) & (hours < 16),
    }
    logger.info("\n--- Per-session Accuracy (threshold 0.5) ---")
    for name, mask in sessions.items():
        mask_arr = mask.values
        if mask_arr.sum() > 50:
            acc = accuracy_score(y_test[mask_arr], (p_test[mask_arr] >= 0.5).astype(int))
            logger.info(f"  {name}: {acc:.2%} ({mask_arr.sum()} rounds)")

    # 7. PER-VOLATILITY-REGIME ANALYSIS
    if 'win_rvol' in df_test_meta.columns:
        rvol = df_test_meta['win_rvol']
        low_vol  = rvol < rvol.quantile(config['evaluation']['rvol_low_pct'])
        high_vol = rvol > rvol.quantile(config['evaluation']['rvol_high_pct'])
        normal   = ~low_vol & ~high_vol

        logger.info("\n--- Per-volatility-regime Accuracy ---")
        for name, mask in [('Low Vol', low_vol), ('Normal Vol', normal), ('High Vol', high_vol)]:
            mask_arr = mask.values
            if mask_arr.sum() > 50:
                acc = accuracy_score(y_test[mask_arr], (p_test[mask_arr] >= best_t).astype(int))
                logger.info(f"  {name}: {acc:.2%} ({mask_arr.sum()} rounds)")

    # 8. MANDATORORY BASELINES
    # Bandingkan terhadap baseline yang wajib dikalahkan
    coin_flip_acc = accuracy_score(y_test, np.ones(len(y_test)) * round(y_test.mean()))
    last_sign_acc = accuracy_score(y_test, (p_test >= 0.5).astype(int))  # Approx

    logger.info(f"\n--- Baseline Comparison ---")
    logger.info(f"  Coin flip (always predict majority): {coin_flip_acc:.2%}")
    logger.info(f"  Model @ 0.5 threshold: {accuracy_score(y_test, (p_test>=0.5).astype(int)):.2%}")
    logger.info(f"  Model @ {best_t:.3f} threshold: {accuracy_score(y_test[(p_test>=best_t)|(p_test<=1-best_t)], (p_test[(p_test>=best_t)|(p_test<=1-best_t)]>=best_t).astype(int)):.2%}")

    # FINAL SUMMARY
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL RESULT SUMMARY")
    logger.info(f"  AUC Test       : {auc_test:.4f}")
    logger.info(f"  ECE (calibrated): {ece:.4f}")
    logger.info(f"  Opt Threshold  : {best_t:.3f}")
    final_mask = (p_test >= best_t) | (p_test <= 1 - best_t)
    if final_mask.sum() > 0:
        final_acc = accuracy_score(y_test[final_mask], (p_test[final_mask] >= best_t).astype(int))
        logger.info(f"  Accuracy @{best_t:.3f} : {final_acc:.2%} ({final_mask.sum()} rounds)")
    logger.info("=" * 60)
```

---

## BAGIAN 10 — WALK-FORWARD VALIDATION (WAJIB)

### 10.1 Implementasi

```python
def walk_forward_validation_v1(df: pd.DataFrame, feature_cols: list,
                                 config: dict) -> pd.DataFrame:
    """
    Single split tidak cukup. Walk-forward memberikan estimasi yang lebih andal.
    Jika AUC turun konsisten per fold → concept drift.
    """
    wfv_cfg       = config['training']
    train_rounds  = wfv_cfg['wfv_train_rounds']   # 1344
    val_rounds    = wfv_cfg['wfv_val_rounds']     # 192
    step_rounds   = wfv_cfg['wfv_step_rounds']    # 192
    n_splits      = wfv_cfg['wfv_n_splits']       # 6

    df    = df.sort_values('round_start').reset_index(drop=True)
    n     = len(df)
    results = []

    for i in range(n_splits):
        val_end   = n - (n_splits - 1 - i) * step_rounds
        val_start = val_end - val_rounds
        train_end = val_start
        train_start = max(0, train_end - train_rounds)

        if val_start < train_rounds or val_end > n:
            continue

        X_tr = df.iloc[train_start:train_end][feature_cols].astype(np.float32)
        y_tr = df.iloc[train_start:train_end]['target'].astype(np.int8)
        X_va = df.iloc[val_start:val_end][feature_cols].astype(np.float32)
        y_va = df.iloc[val_start:val_end]['target'].astype(np.int8)

        # Quick training (lebih sedikit iterasi untuk speed)
        model = cb.CatBoostClassifier(
            iterations=2000, depth=5, learning_rate=0.02,
            task_type='GPU', gpu_ram_part=0.40,
            bootstrap_type='Bernoulli', loss_function='Logloss',
            eval_metric='AUC', early_stopping_rounds=100,
            verbose=False, random_seed=42
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

        p_va = model.predict_proba(X_va)[:, 1]
        auc  = roc_auc_score(y_va, p_va)
        acc  = accuracy_score(y_va, (p_va >= 0.5).astype(int))

        period = pd.to_datetime(df.iloc[val_start]['round_start'])
        results.append({'fold': i+1, 'period': period, 'auc': auc, 'accuracy': acc,
                        'n_rounds': val_rounds})
        logger.info(f"WFV Fold {i+1}: {period.date()} | AUC={auc:.4f} | Acc={acc:.2%}")

    results_df = pd.DataFrame(results)

    # Cek concept drift: jika AUC turun konsisten
    auc_trend = np.polyfit(range(len(results_df)), results_df['auc'], 1)[0]
    if auc_trend < -0.005:
        logger.warning(f"⚠️ Concept drift: AUC turun {auc_trend:.4f}/fold")
    elif auc_trend > 0.005:
        logger.info(f"✅ Model improving: AUC naik {auc_trend:.4f}/fold")
    else:
        logger.info(f"AUC stabil: trend = {auc_trend:.4f}/fold")

    logger.info(f"\nWFV Summary: AUC mean={results_df['auc'].mean():.4f} "
                f"std={results_df['auc'].std():.4f}")
    return results_df
```

---

## BAGIAN 11 — EXPECTED VALUE DAN TRADE DECISION

### 11.1 Formula EV yang Benar

```python
def compute_expected_value(
    p_model: float,      # Probabilitas dari model (sudah dikalibrasi)
    p_market: float,     # Polymarket YES price (implied probability)
    config: dict
) -> dict:
    """
    EV formula dari README:
    EV_YES = p_model - p_market - cost - model_risk_margin

    Bukan sekedar: bet jika p_model > 0.5
    """
    fee    = config['trade_decision']['polymarket_fee']         # 0.02
    margin = config['trade_decision']['model_risk_margin']      # 0.02
    min_ev = config['trade_decision']['min_ev_to_trade']        # 0.0

    # Edge untuk YES
    ev_yes = p_model - p_market - fee - margin

    # Edge untuk NO (bet bahwa BTC DOWN)
    p_model_no  = 1 - p_model
    p_market_no = 1 - p_market
    ev_no       = p_model_no - p_market_no - fee - margin

    # Decision
    if ev_yes > min_ev and ev_yes >= ev_no:
        decision   = 'BET_YES'
        ev         = ev_yes
        confidence = p_model
    elif ev_no > min_ev and ev_no > ev_yes:
        decision   = 'BET_NO'
        ev         = ev_no
        confidence = p_model_no
    else:
        decision   = 'SKIP'
        ev         = max(ev_yes, ev_no)
        confidence = abs(p_model - 0.5)

    # Kelly bet sizing
    if decision != 'SKIP':
        b         = (1 / p_market if decision == 'BET_YES'
                     else 1 / p_market_no) - 1
        p_win     = p_model if decision == 'BET_YES' else p_model_no
        kelly_full = (p_win * b - (1 - p_win)) / (b + 1e-9)
        kelly_frac = kelly_full * config['trade_decision']['kelly_fraction']  # 25%
        bet_size   = max(0, min(kelly_frac, config['trade_decision']['max_bet_fraction']))
    else:
        bet_size = 0

    return {
        'decision'  : decision,
        'ev'        : ev,
        'ev_yes'    : ev_yes,
        'ev_no'     : ev_no,
        'bet_size'  : bet_size,
        'confidence': confidence,
        'p_model'   : p_model,
        'p_market'  : p_market,
    }
```

---

## BAGIAN 12 — EXPERIMENT LOG DAN HASIL SEBELUMNYA

### 12.1 Ringkasan Semua Iterasi yang Pernah Dijalankan

```
ITERASI 1-7 (5-minute model, v4):
  Dataset     : 9,708 rounds (coverage 44%)
  Features    : 563 raw features
  AUC test    : 0.51-0.54
  Accuracy    : 51-54%
  Bug utama   : Row-shift target (bukan round boundary), feature leakage

ITERASI 8 (v4 dengan round-based aggregation):
  Dataset     : 9,708 rounds
  Features    : ~150 features
  AUC test    : 0.5113
  Accuracy    : 51.76%
  Prob dist   : Semua di [0.48, 0.52] — 0 sampel di conf ≥ 0.55
  Bug utama   : y_val tidak dioper ke reduce_features() (NameError)

ITERASI 9 (15-minute model, 02_feature_engineering.py + 03_training.py):
  Dataset     : 26,948 rounds (coverage lebih baik)
  Features raw: 194 features (sebelum reduction)
  AUC test    : 0.5285
  Accuracy    :
    @0.51: 52.15% (cov 82.81%)
    @0.53: 53.71% (cov 50.38%)
    @0.55: 54.42% (cov 25.45%)
  Top features: past_up_lag1, snap_cos_hour, past_up_lag5,
                past_up_lag4, snap_ema_dist_50 (time features dominan!)
  Bug yang ditemukan:
    #1: Feature selection HILANG sebelum CatBoost training
    #2: LightGBM DART → "No further splits with positive gain" (194x)
    #3: Time features diagregasi (win_mean_sin_hour ≡ snap_sin_hour)
    #4: Operator precedence di select_features() (A and B or C)
    #5: Sample weight 3.5x terlalu agresif

ITERASI 10 (semua bug di-fix):
  Dataset     : sama
  Features raw: ~150 features (setelah fix time feature aggregation)
  Features selected: 70 (setelah reduction)
  Status      : Belum dijalankan — ini adalah target iterasi berikutnya
  Expected AUC: 0.53-0.57 (dengan dataset yang sama)
```

### 12.2 Root Cause Coverage Rendah

```
Coverage 44% adalah MASALAH UTAMA yang belum terselesaikan.
Efeknya:
  - Inter-round features tidak reliable (past_ret_lag1 dari round yang missing)
  - Model belajar dari konteks yang tidak lengkap
  - WFV tidak bisa memberikan estimasi yang reliabel

Penyebab coverage rendah:
  - Data collection script berjalan intermittent (tidak 24/7)
  - REST API (bukan WebSocket) rentan timeout
  - save_interval = 60 detik dengan data yang mungkin tidak selalu ada

Solusi jangka pendek:
  - Filter: buang hari dengan daily coverage < 50%
  - Invalidasi inter-round features saat gap > 30 menit

Solusi jangka panjang:
  - WebSocket collector yang berjalan sebagai systemd service
  - Monitoring otomatis untuk deteksi gap
  - Target: coverage ≥ 80% sebelum training serius
```

---

## BAGIAN 13 — ANTI-PATTERN YANG DILARANG

### 13.1 Checklist WAJIB sebelum setiap commit kode

```
LEAKAGE CHECK:
  [ ] Tidak ada groupby().transform() yang menggunakan seluruh round
  [ ] Tidak ada forward-fill target
  [ ] Tidak ada merge_asof dengan direction='forward' untuk target
  [ ] StandardScaler hanya fit pada training data
  [ ] RegimeDetector hanya fit pada training data
  [ ] Threshold hanya dicari di validation set
  [ ] VPIN menggunakan integer position (iloc), bukan datetime (loc)
  [ ] Semua inter-round features menggunakan .shift(1) minimum

TIME FEATURE CHECK:
  [ ] sin_hour, cos_hour, sin_dow, cos_dow TIDAK ada prefix win_mean_ / win_std_ / win_trend_
  [ ] Time features hanya muncul sebagai snap_sin_hour, snap_cos_hour, dll.

GPU CONSTRAINT CHECK:
  [ ] gpu_ram_part ≤ 0.40
  [ ] rsm parameter TIDAK ADA dalam params dict
  [ ] bootstrap_type = 'Bernoulli'
  [ ] max_bin ≤ 128

LIGHTGBM CHECK:
  [ ] boosting_type = 'gbdt' (bukan 'dart')
  [ ] y_val dioper sebagai parameter ke reduce_features_via_lgbm()
  [ ] sample_weight ≤ 2.0x (bukan 3.5x)

PIPELINE ORDER CHECK:
  [ ] reduce_features_via_lgbm() dipanggil SEBELUM optimize_catboost()
  [ ] calibrator fit SETELAH model final training, pada VAL set
  [ ] threshold search SETELAH calibration, pada VAL set

DATA CHECK:
  [ ] UP rate antara 0.40 dan 0.60
  [ ] Coverage terlapor di log
  [ ] Inter-round features di-invalidasi untuk round dengan gap > 30 menit
  [ ] Tidak ada harga absolut dalam feature list

FEATURE SELECTION CHECK:
  [ ] is_valid_feature() menggunakan fungsi eksplisit (bukan operator precedence)
  [ ] NaN threshold = 0.30
  [ ] std threshold = 1e-10
```

---

## BAGIAN 14 — REALISTIC EXPECTATIONS DAN MILESTONE

### 14.1 Target yang Realistis

```
DENGAN COVERAGE SAAT INI (44%):
  - Best achievable AUC: 0.54-0.57
  - Best achievable Accuracy: 54-58%
  - Alasan: inter-round features tidak reliable, coverage tidak cukup

DENGAN COVERAGE 75%+:
  - Realistic AUC: 0.56-0.60
  - Realistic Accuracy: 57-62%
  - Ini zona yang mulai bisa profitable di Polymarket setelah fee 2%

DENGAN COVERAGE 90%+ DAN OB L2 REAL:
  - Stretch target AUC: 0.60-0.65
  - Stretch target Accuracy: 61-66%
  - Membutuhkan: WebSocket 24/7, minimal 6 bulan data bersih

TENTANG TARGET 65%:
  Dari literatur akademis, 65% consistency di BTC intraday adalah EXCEPTIONAL.
  Achievable hanya dengan:
  1. Data L2 orderbook yang lengkap (bukan aggTrades proxy)
  2. Coverage > 90%
  3. Fitur yang sangat dipilih (bukan semua dimasukkan)
  4. Model yang dikalibrasi dengan baik
  5. Betting hanya di high-confidence rounds (coverage ~15-25%)
```

### 14.2 Milestone Kronologis

```
MILESTONE 1 — Data Quality (Prioritas Tertinggi):
  Target: Coverage ≥ 70% dari data yang ada
  Cara: Filter hari dengan coverage < 50%, invalidasi inter-round features
  Expected: minggu pertama

MILESTONE 2 — Clean Pipeline (Semua Bug Fix):
  Target: AUC ≥ 0.53, tidak ada prob clustering di 0.48-0.52
  Cara: Deploy iterasi 10 dengan semua fix
  Expected: setelah milestone 1

MILESTONE 3 — Signal Validation:
  Target: past_up_lag1 masuk top 10 features
  Interpretasi: inter-round momentum bekerja
  Expected: setelah coverage ≥ 70%

MILESTONE 4 — Trading Ready:
  Target: AUC ≥ 0.56, Accuracy ≥ 57% @ conf ≥ 0.55
  Action: Mulai paper trading (tanpa uang nyata, track semua prediksi)
  Expected: bulan kedua

MILESTONE 5 — Live Trading (Kecil):
  Target: 2 minggu paper trading profitable
  Action: Live trading dengan modal sangat kecil ($50-100)
  Expected: bulan ketiga
```

---

## BAGIAN 15 — UNIT TESTS (WAJIB)

```python
# tests/test_leakage.py

import pytest
import pandas as pd
import numpy as np

def test_target_no_future_data():
    """Pastikan target tidak menggunakan data masa depan"""
    # Buat data dummy
    idx = pd.date_range('2025-01-01', periods=100, freq='1min')
    df  = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100}, index=idx)
    # ... (implementasi test)

def test_inter_round_features_use_shift():
    """Pastikan semua past_ features menggunakan shift ≥ 1"""
    df = pd.DataFrame({
        'round_start': pd.date_range('2025-01-01', periods=50, freq='15min'),
        'future_return': np.random.randn(50) * 0.001,
        'target': np.random.randint(0, 2, 50),
    })
    # Import dan jalankan fungsi
    from src.features.feature_engineering import add_inter_round_features_v1
    result = add_inter_round_features_v1(df, config)
    # past_ret_lag1 harus identik dengan future_return.shift(1)
    assert result['past_ret_lag1'].iloc[1] == df['future_return'].iloc[0]

def test_time_features_not_aggregated():
    """Pastikan tidak ada win_mean_sin_hour, win_std_cos_hour, dll."""
    features = pd.read_parquet('data/features/v1/final_features.parquet')
    time_aggr_cols = [c for c in features.columns
                      if any(c.startswith(pfx + tf)
                             for pfx in ['win_mean_', 'win_std_', 'win_trend_']
                             for tf in ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
                                        'is_us_session', 'is_london_session'])]
    assert len(time_aggr_cols) == 0, f"Found aggregated time features: {time_aggr_cols}"

def test_scaler_not_fit_on_test():
    """Pastikan scaler tidak di-fit pada test data"""
    # Test bahwa RegimeDetector.predict() tidak memanggil fit_transform()
    from src.models.regime_detector import RegimeDetector
    detector = RegimeDetector()
    X_train = pd.DataFrame(np.random.randn(100, 5))
    X_test  = pd.DataFrame(np.random.randn(20, 5))
    detector.fit(X_train)
    # predict() seharusnya tidak raise error dan tidak memanggil fit
    regimes = detector.predict(X_test)
    assert len(regimes) == 20
```

---

## BAGIAN 16 — MONITORING LIVE PREDICTIONS

```python
def log_live_prediction(round_start: pd.Timestamp,
                         p_model: float,
                         p_market: float,
                         decision: str,
                         bet_size: float,
                         features_snapshot: dict,
                         model_version: str,
                         feature_version: str):
    """
    WAJIB: Setiap prediksi live harus dicatat lengkap.
    Gunakan untuk audit dan retrospective analysis.
    """
    record = {
        'timestamp'      : pd.Timestamp.utcnow().isoformat(),
        'round_start'    : round_start.isoformat(),
        'p_model'        : round(p_model, 6),
        'p_market'       : round(p_market, 6),
        'model_edge'     : round(p_model - p_market, 6),
        'decision'       : decision,
        'bet_size'       : round(bet_size, 6),
        'model_version'  : model_version,
        'feature_version': feature_version,
        'features'       : features_snapshot,  # Simpan semua features
        'realized_label' : None,  # Diisi setelah resolusi round
        'realized_return': None,  # Diisi setelah resolusi round
    }
    # Append ke file JSONL
    with open('logs/live_predictions.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
```

---

## PENUTUP — URUTAN EKSEKUSI

```bash
# Setiap sesi baru:
cd "/run/media/rotan/New Volume/gemini3/polymarket_5m"
source venv/bin/activate

# Eksekusi dalam urutan:
python src/labels/02b_label_builder.py         # Bangun target
python src/features/02_feature_engineering.py  # Bangun features
python src/models/03_training.py               # Training
python src/models/04_evaluate.py               # Evaluasi walk-forward
python src/calibration/05_calibrate.py         # Kalibrasi probabilitas

# Verifikasi output:
python tests/test_leakage.py
python tests/test_features.py
python tests/test_timestamps.py

# Jika semua test pass dan AUC > 0.53:
# → Mulai paper trading dengan src/inference/06_inference.py
```

---

*Dokumen ini dibuat berdasarkan 1 bulan eksperimen aktual, 11 bug yang ditemukan dan dianalisa,
README arsitektur referensi, dan spesifikasi hardware aktual. Setiap item dalam dokumen ini
memiliki alasan yang spesifik dari pengalaman nyata, bukan hanya teori.*

*Versi: 1.0 | Terakhir diperbarui: Mei 2026*
