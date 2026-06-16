import pandas as pd
import numpy as np
from pathlib import Path

# Konfigurasi Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/features/v1/final_features.parquet"

def inspect_features():
    if not FEATURES_PATH.exists():
        print(f"❌ File tidak ditemukan di: {FEATURES_PATH}")
        return

    print(f"🔍 Memuat data dari: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    
    # 1. Informasi Dasar
    print("\n" + "="*60)
    print(f"📊 INFORMASI DASAR")
    print("="*60)
    print(f"Jumlah Baris (Rounds) : {len(df):,}")
    print(f"Jumlah Kolom (Fitur)  : {len(df.columns)}")
    
    if 'round_start' in df.columns:
        print(f"Rentang Waktu        : {df['round_start'].min()} s/d {df['round_start'].max()}")

    # 2. Pengelompokkan Fitur
    print("\n" + "="*60)
    print(f"📂 KATEGORI FITUR")
    print("="*60)
    categories = {
        "Microstructure (Snapshot)": [c for c in df.columns if c.startswith('snap_')],
        "Window Mean (15m)": [c for c in df.columns if c.startswith('win_mean_')],
        "Window Std (15m)": [c for c in df.columns if c.startswith('win_std_')],
        "Window Trend (15m)": [c for c in df.columns if c.startswith('win_trend_')],
        "Inter-round Lags": [c for c in df.columns if c.startswith('past_')],
        "Target/Metadata": [c for c in ['target', 'future_return', 'round_start'] if c in df.columns]
    }
    
    for cat, cols in categories.items():
        print(f"{cat:<30}: {len(cols)} kolom")

    # 3. Analisis Kualitas Data (Top 10 High Nulls)
    print("\n" + "="*60)
    print(f"⚠️ ANALISIS KUALITAS (Missing & Constant)")
    print("="*60)
    null_counts = df.isnull().sum().sort_values(ascending=False)
    if null_counts.max() > 0:
        print("Kolom dengan nilai NaN terbanyak:")
        print(null_counts.head(10))
    else:
        print("✅ Tidak ada nilai NaN dalam dataset.")

    # Cek fitur konstan (std = 0)
    numeric_df = df.select_dtypes(include=[np.number])
    const_cols = numeric_df.columns[numeric_df.std() < 1e-12].tolist()
    if const_cols:
        print(f"\nDitemukan {len(const_cols)} fitur konstan (std ≈ 0):")
        print(const_cols[:10])
    else:
        print("\n✅ Tidak ada fitur konstan.")

    # 4. Korelasi dengan Target (Top 20)
    if 'target' in df.columns:
        print("\n" + "="*60)
        print(f"🎯 KORELASI TERKUAT DENGAN TARGET")
        print("="*60)
        correlations = numeric_df.corr()['target'].abs().sort_values(ascending=False)
        # Hapus target itu sendiri dan metadata dari list korelasi
        to_drop = ['target', 'future_return', 'is_win', 'net_profit', 'bankroll']
        correlations = correlations.drop([c for c in to_drop if c in correlations.index], errors='ignore')
        print(correlations.head(20))

    # 5. List Nama Fitur (Alphabetical)
    print("\n" + "="*60)
    print(f"📋 DAFTAR SEMUA FITUR (50 pertama)")
    print("="*60)
    all_features = sorted([c for c in df.columns if not any(x in c for x in ['target', 'round_start', 'return'])])
    for i in range(0, min(len(all_features), 50), 2):
        row = all_features[i:i+2]
        print(f"{row[0]:<40} {row[1] if len(row)>1 else ''}")

if __name__ == "__main__":
    inspect_features()
