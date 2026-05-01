import pandas as pd
import numpy as np
import talib
import logging
import yaml
import os
import json
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    SEMUA technical features harus STASIONER (bukan harga absolut)
    Menggunakan TA-Lib
    """
    res = df.copy()
    close = res['close'].values
    high = res['high'].values
    low = res['low'].values
    volume = res['volume'].values

    # 1. RSI
    res['rsi_7']  = talib.RSI(close, timeperiod=7)
    res['rsi_14'] = talib.RSI(close, timeperiod=14)
    res['rsi_21'] = talib.RSI(close, timeperiod=21)

    # 2. MACD
    macd, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    res['macd'] = macd
    res['macd_hist'] = macd_hist

    # 3. Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    res['bb_pct_20']   = (close - lower) / (upper - lower + 1e-9)
    res['bb_width_20'] = (upper - lower) / (middle + 1e-9)

    # 4. ATR (normalized)
    res['atr_14_norm'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-9)

    # 5. EMA Distance (normalized deviation)
    for p in [9, 21, 50]:
        ema = talib.EMA(close, timeperiod=p)
        res[f'ema_dist_{p}'] = (close - ema) / (ema + 1e-9)

    # 6. Volume Ratio
    res['vol_ratio_15'] = volume / (pd.Series(volume).rolling(15).mean().values + 1e-9)
    res['vol_ratio_60'] = volume / (pd.Series(volume).rolling(60).mean().values + 1e-9)

    # 7. Rolling VWAP Deviation
    pv = close * volume
    res['vwap_rolling_60'] = pd.Series(pv).rolling(60).sum().values / (pd.Series(volume).rolling(60).sum().values + 1e-9)
    res['vwap_dev'] = (close - res['vwap_rolling_60']) / (res['vwap_rolling_60'] + 1e-9)
    res['vwap_dev_norm'] = res['vwap_dev'] / (res['atr_14_norm'] + 1e-9)

    # 8. Price Velocity
    log_ret = np.log(close / (pd.Series(close).shift(1).values + 1e-9))
    res['log_ret'] = log_ret
    res['price_velocity'] = log_ret - pd.Series(log_ret).shift(3).values

    # 9. Momentum multi-timeframe
    res['momentum_1h'] = close / (pd.Series(close).shift(60).values + 1e-9) - 1
    res['momentum_4h'] = close / (pd.Series(close).shift(240).values + 1e-9) - 1

    # 10. Realized Volatility
    for window in [60, 180, 300]:
        n_candles = window // 60
        if n_candles < 1: n_candles = 1
        res[f'rvol_{window}'] = pd.Series(log_ret).rolling(n_candles).std().values * np.sqrt(n_candles)

    return res

def add_time_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    ATURAN KERAS: TIME FEATURES TIDAK BOLEH DIAGREGASI
    Hanya ditambahkan ke DataFrame tick/1m, nanti diambil snapshot-nya.
    """
    timestamps = df.index
    hours = timestamps.hour + timestamps.minute / 60.0
    dows  = timestamps.dayofweek

    df['sin_hour'] = np.sin(2 * np.pi * hours / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hours / 24)
    df['sin_dow']  = np.sin(2 * np.pi * dows / 7)
    df['cos_dow']  = np.cos(2 * np.pi * dows / 7)

    slot_15m = timestamps.hour * 4 + timestamps.minute // 15
    df['sin_15m'] = np.sin(2 * np.pi * slot_15m / 96)
    df['cos_15m'] = np.cos(2 * np.pi * slot_15m / 96)

    df['is_us_session']     = ((timestamps.hour >= 13) & (timestamps.hour < 21)).astype(float)
    df['is_london_session'] = ((timestamps.hour >= 7)  & (timestamps.hour < 16)).astype(float)
    df['is_overlap']        = df['is_us_session'] * df['is_london_session']
    df['is_weekend']        = (dows >= 5).astype(float)

    df['is_roll_hour']       = (timestamps.hour % 8 == 0).astype(float)
    df['minutes_since_roll'] = ((timestamps.hour % 8) * 60 + timestamps.minute).astype(float)

    return df

def compute_vpin_correct(df_trades, buckets=30):
    """
    VPIN: Detect informed trading from trade flow
    """
    if df_trades.empty: return pd.DataFrame(columns=['vpin'])
    df = df_trades.copy()
    
    # Ensure numeric types
    df['qty'] = df['qty'].astype(float)
    if df['is_buyer_maker'].dtype == object:
        df['is_buyer_maker'] = df['is_buyer_maker'].map({'True': True, 'False': False, True: True, False: False})
    
    df['buy_vol']  = np.where(df['is_buyer_maker'] == False, df['qty'], 0)
    df['sell_vol'] = np.where(df['is_buyer_maker'] == True,  df['qty'], 0)
    df['total_vol'] = df['buy_vol'] + df['sell_vol']
    df['imbalance'] = (df['buy_vol'] - df['sell_vol']).abs()

    cum_vol    = df['total_vol'].cumsum().values
    total_vol  = cum_vol[-1]
    bucket_size = total_vol / buckets

    vpin_vals = np.zeros(len(df))
    prev_pos  = 0

    for i in range(buckets):
        target_vol = (i + 1) * bucket_size
        next_pos_arr = np.where(cum_vol >= target_vol)[0]
        curr_pos = next_pos_arr[0] if len(next_pos_arr) > 0 else len(df) - 1

        bucket = df.iloc[prev_pos:curr_pos + 1]
        bucket_buy  = bucket['buy_vol'].sum()
        bucket_sell = bucket['sell_vol'].sum()
        bucket_vol  = bucket['total_vol'].sum()
        
        # CORRECT VPIN: abs(Sum Buy - Sum Sell) / Sum Total
        vpin_i = abs(bucket_buy - bucket_sell) / (bucket_vol + 1e-9)
        vpin_vals[prev_pos:curr_pos + 1] = vpin_i
        prev_pos = curr_pos + 1

    df['vpin'] = vpin_vals
    return df[['vpin']]

def compute_obi_v1(df_ob: pd.DataFrame) -> pd.DataFrame:
    """
    Order Book Imbalance: Selisih volume bid vs ask di top levels.
    df_ob memiliki kolom bids/asks yang merupakan STRINGS dari list of lists.
    """
    if df_ob.empty: return pd.DataFrame(columns=['obi'])
    
    def parse_l1_vol(x):
        try:
            data = json.loads(x)
            return float(data[0][1])
        except: return 0.0

    b0_vol = df_ob['bids'].apply(parse_l1_vol)
    a0_vol = df_ob['asks'].apply(parse_l1_vol)
    
    obi = (b0_vol - a0_vol) / (b0_vol + a0_vol + 1e-9)
    return pd.DataFrame({'obi': obi}, index=df_ob.index)

def compute_ofi_v1(df_ob: pd.DataFrame) -> pd.DataFrame:
    """
    Order Flow Imbalance: Perubahan liquidity pada bid/ask levels antar snapshot.
    """
    if df_ob.empty: return pd.DataFrame(columns=['ofi'])
    
    def parse_l1_data(x):
        try:
            data = json.loads(x)
            return float(data[0][0]), float(data[0][1])
        except: return 0.0, 0.0

    b_data = df_ob['bids'].apply(parse_l1_data)
    a_data = df_ob['asks'].apply(parse_l1_data)
    
    b_price = b_data.apply(lambda x: x[0])
    b_vol   = b_data.apply(lambda x: x[1])
    a_price = a_data.apply(lambda x: x[0])
    a_vol   = a_data.apply(lambda x: x[1])
    
    # Delta Bid
    db = np.where(b_price > b_price.shift(1), b_vol,
         np.where(b_price < b_price.shift(1), -b_vol.shift(1),
         b_vol - b_vol.shift(1)))
    
    # Delta Ask
    da = np.where(a_price < a_price.shift(1), a_vol,
         np.where(a_price > a_price.shift(1), -a_vol.shift(1),
         a_vol - a_vol.shift(1)))
    
    ofi = pd.Series(db - da, index=df_ob.index).fillna(0)
    ofi_norm = ofi / (b_vol + a_vol + 1e-9)
    
    return pd.DataFrame({'ofi': ofi_norm}, index=df_ob.index)

def aggregate_window_features_v1(df_full: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Input: df_full dengan index datetime
    Output: 1 baris per round, index = round_start timestamp
    """
    time_like_cols = set(config['features']['time_like_cols'])

    df = df_full.copy()
    # Label setiap tick dengan round yang akan diprediksi (round T dimulai pada T)
    # Fitur diambil dari [T-15m, T)
    df['round_label'] = df.index.floor('15min')

    all_cols      = [c for c in df.columns if c != 'round_label']
    signal_cols   = [c for c in all_cols if c not in time_like_cols]
    
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

    # Price Action (dari harga mentah)
    if 'close' in df.columns:
        price_grp   = grp['close']
        price_first = price_grp.first()
        price_last  = price_grp.last()
        price_min   = price_grp.min()
        price_max   = price_grp.max()

        win_ret = pd.DataFrame({
            'win_return'         : price_last / (price_first + 1e-9) - 1,
            'win_close_position' : (price_last - price_min) / (price_max - price_min + 1e-9),
        }, index=price_first.index)
        parts.append(win_ret)

    # Gabungkan semua parts sekaligus
    df_aggr = pd.concat(parts, axis=1)

    # GROUP 3: Acceleration (Window half comparison)
    # Dihitung dari df_full karena butuh resolusi tick/1m
    for col in ['rsi_14', 'ema_dist_21', 'vwap_dev_norm']:
        if col in df.columns:
            df['tick_idx'] = df.groupby('round_label').cumcount()
            df['win_size'] = df.groupby('round_label')['tick_idx'].transform('max')
            df['is_late']  = df['tick_idx'] > (df['win_size'] / 2)
            
            accel = df.groupby(['round_label', 'is_late'])[col].mean().unstack()
            if accel.shape[1] == 2:
                accel.columns = ['early', 'late']
                accel_series = (accel['late'] - accel['early']).rename(f'win_{col}_accel')
                df_aggr = pd.concat([df_aggr, accel_series], axis=1)

    # GROUP 5: Cross-Source Interaction (Snapshot only for now)
    if 'snap_rsi_14' in df_aggr.columns:
        # RSI divergence proxy (not real divergence, but interaction)
        # Assuming we had OFI, but for now we only have technicals
        pass

    # SHIFT INDEX: features dari [T-15m, T) → digunakan untuk prediksi round T
    df_aggr.index = df_aggr.index + pd.Timedelta(minutes=15)
    df_aggr.index.name = 'round_start'

    return df_aggr

def add_inter_round_features_v1(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    WAJIB: Semua menggunakan .shift(N) dengan N >= 1
    DILARANG: .shift(0) = current round = leakage
    """
    res = df.copy().sort_index()
    n_lags   = config['features']['inter_round_lags']
    roll_wins = config['features']['inter_round_rolling']

    # Lag returns dan outcomes (asumsi df sudah punya 'future_return' dan 'target' dari join)
    if 'future_return' in res.columns:
        for lag in range(1, n_lags + 1):
            res[f'past_ret_lag{lag}'] = res['future_return'].shift(lag)
        res['past_ret_abs_lag1'] = res['future_return'].shift(1).abs()

    if 'target' in res.columns:
        for lag in range(1, n_lags + 1):
            res[f'past_up_lag{lag}'] = res['target'].shift(lag)

    # Rolling stats antar-round
    for w in roll_wins:
        if 'future_return' in res.columns:
            past_ret = res['future_return'].shift(1)
            res[f'past_ret_sum_{w}']  = past_ret.rolling(w, min_periods=max(1, w//2)).sum()
            res[f'past_rvol_{w}']     = past_ret.rolling(w, min_periods=max(1, w//2)).std()
        
        if 'target' in res.columns:
            past_up  = res['target'].shift(1)
            res[f'past_up_rate_{w}']  = past_up.rolling(w, min_periods=max(1, w//2)).mean()

    # Mean reversion z-score
    if 'past_ret_lag1' in res.columns and 'past_rvol_8' in res.columns:
        res['past_ret_zscore_8'] = res['past_ret_lag1'] / (res['past_rvol_8'] + 1e-9)

    return res

if __name__ == "__main__":
    # Load config
    with open("/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load 1m Klines (Base Dataset)
    klines_path = config['paths']['data']['klines_1m']
    logger.info(f"Loading klines from {klines_path}...")
    df_1m = pd.read_parquet(klines_path)
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m = df_1m.set_index('timestamp')
    df_1m = df_1m.sort_index()

    # 2. Add technical and time features to 1m data
    logger.info("Adding technical & time features...")
    df_1m = add_technical_features_v1(df_1m)
    df_1m = add_time_features_v1(df_1m)

from joblib import Parallel, delayed

def process_day_features(day, config, day_index):
    """
    Worker function untuk memproses data mikrostruktur harian secara paralel.
    """
    day_str = day.strftime('%Y-%m-%d')
    
    # Paths
    ob_path = Path(config['paths']['data']['orderbook']) / f"{day_str}_BTCUSDT_ob200.parquet"
    tr_path = Path(config['paths']['data']['aggtrades']) / f"BTCUSDT-aggTrades-{day_str}.parquet"
    
    day_micro = pd.DataFrame(index=day_index)
    
    # A. Orderbook Features (OBI, OFI)
    if ob_path.exists():
        try:
            df_ob = pd.read_parquet(ob_path)
            df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp_ms'], unit='ms')
            df_ob = df_ob.set_index('timestamp')
            
            df_obi = compute_obi_v1(df_ob)
            df_ofi = compute_ofi_v1(df_ob)
            
            # Resample to 1m
            micro_ob = pd.concat([df_obi, df_ofi], axis=1).resample('1min').agg(['mean', 'std'])
            micro_ob.columns = [f"{c[0]}_{c[1]}" for c in micro_ob.columns]
            day_micro = day_micro.join(micro_ob, how='left')
        except Exception as e:
            logger.error(f"Error processing OB for {day_str}: {e}")

    # B. Trade Flow Features (VPIN)
    if tr_path.exists():
        try:
            df_tr = pd.read_parquet(tr_path)
            df_tr['timestamp'] = pd.to_datetime(df_tr['timestamp'], unit='ms')
            df_tr = df_tr.set_index('timestamp')
            
            df_vpin = compute_vpin_correct(df_tr)
            
            # Resample to 1m
            micro_tr = df_vpin.resample('1min').agg(['mean', 'std'])
            micro_tr.columns = [f"{c[0]}_{c[1]}" for c in micro_tr.columns]
            day_micro = day_micro.join(micro_tr, how='left')
        except Exception as e:
            logger.error(f"Error processing Trades for {day_str}: {e}")

    return day_micro

if __name__ == "__main__":
    # Load config
    with open("/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load 1m Klines (Base Dataset)
    klines_path = config['paths']['data']['klines_1m']
    logger.info(f"Loading klines from {klines_path}...")
    df_1m = pd.read_parquet(klines_path)
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m = df_1m.set_index('timestamp')
    df_1m = df_1m.sort_index()

    # 2. Add technical and time features to 1m data
    logger.info("Adding technical & time features...")
    df_1m = add_technical_features_v1(df_1m)
    df_1m = add_time_features_v1(df_1m)

    # 3. Process Microstructure Data in Parallel (10 CORES)
    all_days = df_1m.index.normalize().unique()
    logger.info(f"Processing microstructure for {len(all_days)} days using 10 cores...")
    
    # Persiapkan task untuk tiap hari
    tasks = []
    for day in all_days:
        day_str = day.strftime('%Y-%m-%d')
        day_index = df_1m.loc[day_str].index if day_str in df_1m.index else pd.Index([])
        tasks.append((day, config, day_index))
    
    # Jalankan paralel
    micro_results = Parallel(n_jobs=10)(
        delayed(process_day_features)(d, c, idx) for d, c, idx in tqdm(tasks)
    )

    # Merge all microstructure features
    micro_parts = [res for res in micro_results if not res.empty]
    if micro_parts:
        df_micro_full = pd.concat(micro_parts).sort_index()
        # Filter to match df_1m index
        df_micro_full = df_micro_full.reindex(df_1m.index)
        df_1m = pd.concat([df_1m, df_micro_full], axis=1)
        logger.info(f"Microstructure integrated. Columns added: {df_micro_full.columns.tolist()}")
    else:
        logger.warning("No microstructure data found in the specified paths.")

    # 4. Aggregate to 15m windows
    logger.info("Aggregating to 15m windows...")
    df_features = aggregate_window_features_v1(df_1m, config)
    
    # 5. Load targets and join
    targets_path = Path(config['paths']['data']['labels']) / "targets_v1.parquet"
    if targets_path.exists():
        logger.info(f"Loading targets from {targets_path}...")
        df_targets = pd.read_parquet(targets_path)
        df_final = df_features.join(df_targets, how='inner')
        
        # 6. Add inter-round features
        logger.info("Adding inter-round features...")
        df_final = add_inter_round_features_v1(df_final, config)
        
        # Save final features
        output_path = Path(config['paths']['data']['features_v1']) / "final_features.parquet"
        df_final.to_parquet(output_path)
        logger.info(f"Final features saved to {output_path} | Shape: {df_final.shape}")
    else:
        logger.error(f"Targets not found at {targets_path}. Run 02b_label_builder.py first.")

