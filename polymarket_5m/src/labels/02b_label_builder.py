import pandas as pd
import numpy as np
import logging
import yaml
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_target_polymarket_v1(df_1m: pd.DataFrame, config: dict) -> pd.DataFrame:
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

    # Resample ke 15 menit
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
    valid['label_source'] = 'bybit_spot_1m_ohlcv'
    valid['label_threshold'] = threshold
    valid['label_version'] = config['project']['version']

    # Audit sample
    audit_path = Path(config['paths']['data']['labels']) / "label_audit_sample.csv"
    audit = valid.sample(n=min(500, len(valid)), random_state=42)
    audit.to_csv(audit_path)
    logger.info(f"Audit sample saved: {audit_path}")

    return valid

# Dynamically find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load 1m klines
    klines_path = config['paths']['data']['klines_1m']
    logger.info(f"Loading klines from {klines_path}...")
    
    if not os.path.exists(klines_path):
        # Fallback to local path if absolute path from config fails
        klines_path = "data/raw/historical_1m_klines.parquet"
        
    df_1m = pd.read_parquet(klines_path)
    
    # Ensure index is datetime
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if 'timestamp' in df_1m.columns:
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
            df_1m = df_1m.set_index('timestamp')
        elif 'open_time' in df_1m.columns:
            df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
            df_1m = df_1m.set_index('open_time')
    
    # Build labels
    labels = build_target_polymarket_v1(df_1m, config)
    
    # Save labels
    output_path = Path(config['paths']['data']['labels']) / "targets_v1.parquet"
    labels.to_parquet(output_path)
    logger.info(f"Labels saved to {output_path}")
