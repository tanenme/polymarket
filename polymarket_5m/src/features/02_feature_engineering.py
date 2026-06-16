import pandas as pd
import numpy as np
import talib
import logging
import yaml
import os
import json
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def add_technical_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    close = res['close'].values
    high = res['high'].values
    low = res['low'].values
    volume = res['volume'].values

    res['rsi_7']  = talib.RSI(close, timeperiod=7)
    res['rsi_14'] = talib.RSI(close, timeperiod=14)
    res['rsi_21'] = talib.RSI(close, timeperiod=21)

    macd, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    res['macd'] = macd
    res['macd_hist'] = macd_hist

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    res['bb_pct_20']   = (close - lower) / (upper - lower + 1e-9)
    res['bb_width_20'] = (upper - lower) / (middle + 1e-9)

    res['atr_14_norm'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-9)

    for p in [9, 21, 50]:
        ema = talib.EMA(close, timeperiod=p)
        res[f'ema_dist_{p}'] = (close - ema) / (ema + 1e-9)

    res['vol_ratio_15'] = volume / (pd.Series(volume).rolling(15).mean().values + 1e-9)
    res['vol_ratio_60'] = volume / (pd.Series(volume).rolling(60).mean().values + 1e-9)

    pv = close * volume
    res['vwap_rolling_60'] = pd.Series(pv).rolling(60).sum().values / (pd.Series(volume).rolling(60).sum().values + 1e-9)
    res['vwap_dev'] = (close - res['vwap_rolling_60']) / (res['vwap_rolling_60'] + 1e-9)
    res['vwap_dev_norm'] = res['vwap_dev'] / (res['atr_14_norm'] + 1e-9)

    log_ret = np.log(close / (pd.Series(close).shift(1).values + 1e-9))
    res['log_ret'] = log_ret
    res['price_velocity'] = log_ret - pd.Series(log_ret).shift(3).values

    res['momentum_1h'] = close / (pd.Series(close).shift(60).values + 1e-9) - 1
    res['momentum_4h'] = close / (pd.Series(close).shift(240).values + 1e-9) - 1

    for window in [60, 180, 300]:
        n_candles = window // 60
        if n_candles < 1: n_candles = 1
        res[f'rvol_{window}'] = pd.Series(log_ret).rolling(n_candles).std().values * np.sqrt(n_candles)

    return res

def add_time_features_v1(df: pd.DataFrame) -> pd.DataFrame:
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
    if df_trades.empty: return pd.DataFrame(columns=['vpin'])
    df = df_trades.copy()
    df['qty'] = df['qty'].astype(float)
    if df['is_buyer_maker'].dtype == object:
        df['is_buyer_maker'] = df['is_buyer_maker'].map({'True': True, 'False': False, True: True, False: False})
    df['buy_vol']  = np.where(df['is_buyer_maker'] == False, df['qty'], 0)
    df['sell_vol'] = np.where(df['is_buyer_maker'] == True,  df['qty'], 0)
    df['total_vol'] = df['buy_vol'] + df['sell_vol']
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
        vpin_i = abs(bucket_buy - bucket_sell) / (bucket_vol + 1e-9)
        vpin_vals[prev_pos:curr_pos + 1] = vpin_i
        prev_pos = curr_pos + 1
    df['vpin'] = vpin_vals
    return df[['vpin']]

def compute_obi_v1(df_ob: pd.DataFrame) -> pd.DataFrame:
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

def compute_orderbook_features_v2(df_ob: pd.DataFrame) -> pd.DataFrame:
    if df_ob.empty:
        return pd.DataFrame(index=df_ob.index)

    levels = (1, 5, 10, 20, 50)

    def parse_side(raw, reverse=False):
        try:
            data = json.loads(raw)
            parsed = [(float(price), float(size)) for price, size in data if float(size) > 0]
            parsed.sort(key=lambda x: x[0], reverse=reverse)
            return parsed
        except Exception:
            return []

    rows = []
    for bids_raw, asks_raw in zip(df_ob['bids'].values, df_ob['asks'].values):
        bids = parse_side(bids_raw, reverse=True)
        asks = parse_side(asks_raw, reverse=False)

        if not bids or not asks:
            rows.append({})
            continue

        bid_px, bid_qty = bids[0]
        ask_px, ask_qty = asks[0]
        mid = (bid_px + ask_px) / 2.0
        spread = max(ask_px - bid_px, 0.0)
        microprice = (ask_px * bid_qty + bid_px * ask_qty) / (bid_qty + ask_qty + 1e-9)

        row = {
            'spread_bps': spread / (mid + 1e-9) * 10000.0,
            'queue_imb_l1': (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-9),
            'microprice_dev_bps': (microprice - mid) / (mid + 1e-9) * 10000.0,
        }

        for level in levels:
            b = bids[:level]
            a = asks[:level]
            b_depth = sum(q for _, q in b)
            a_depth = sum(q for _, q in a)
            row[f'depth_imb_l{level}'] = (b_depth - a_depth) / (b_depth + a_depth + 1e-9)
            row[f'bid_depth_l{level}'] = b_depth
            row[f'ask_depth_l{level}'] = a_depth
            row[f'wall_bid_l{level}'] = max([q for _, q in b], default=0.0) / (b_depth + 1e-9)
            row[f'wall_ask_l{level}'] = max([q for _, q in a], default=0.0) / (a_depth + 1e-9)

        row['bid_depth_slope_l20'] = sum((mid - px) * q for px, q in bids[:20]) / (sum(q for _, q in bids[:20]) + 1e-9)
        row['ask_depth_slope_l20'] = sum((px - mid) * q for px, q in asks[:20]) / (sum(q for _, q in asks[:20]) + 1e-9)
        row['book_pressure_l20'] = row['depth_imb_l20'] / (row['spread_bps'] + 1e-6)
        rows.append(row)

    res = pd.DataFrame(rows, index=df_ob.index)
    for level in (1, 5, 10, 20):
        col = f'depth_imb_l{level}'
        if col in res.columns:
            res[f'mlofi_proxy_l{level}'] = res[col].diff().fillna(0.0)
    if 'microprice_dev_bps' in res.columns:
        res['microprice_dev_delta'] = res['microprice_dev_bps'].diff().fillna(0.0)
    return res

def compute_ofi_v1(df_ob: pd.DataFrame) -> pd.DataFrame:
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
    db = np.where(b_price > b_price.shift(1), b_vol,
         np.where(b_price < b_price.shift(1), -b_vol.shift(1),
         b_vol - b_vol.shift(1)))
    da = np.where(a_price < a_price.shift(1), a_vol,
         np.where(a_price > a_price.shift(1), -a_vol.shift(1),
         a_vol - a_vol.shift(1)))
    ofi = pd.Series(db - da, index=df_ob.index).fillna(0)
    ofi_norm = ofi / (b_vol + a_vol + 1e-9)
    return pd.DataFrame({'ofi': ofi_norm}, index=df_ob.index)

def compute_taker_flow_features_v2(df_tr: pd.DataFrame) -> pd.DataFrame:
    if df_tr.empty:
        return pd.DataFrame()

    df = df_tr.copy()
    df['qty'] = df['qty'].astype(float)
    df['price'] = df['price'].astype(float)
    if df['is_buyer_maker'].dtype == object:
        df['is_buyer_maker'] = df['is_buyer_maker'].map({'True': True, 'False': False, True: True, False: False})

    taker_buy = df['is_buyer_maker'] == False
    df['notional'] = df['price'] * df['qty']
    df['buy_qty'] = np.where(taker_buy, df['qty'], 0.0)
    df['sell_qty'] = np.where(~taker_buy, df['qty'], 0.0)
    df['buy_notional'] = np.where(taker_buy, df['notional'], 0.0)
    df['sell_notional'] = np.where(~taker_buy, df['notional'], 0.0)
    df['buy_count'] = taker_buy.astype(float)
    df['sell_count'] = (~taker_buy).astype(float)

    large_cut = df['notional'].quantile(0.95)
    large = df['notional'] >= large_cut
    df['large_buy_notional'] = np.where(taker_buy & large, df['notional'], 0.0)
    df['large_sell_notional'] = np.where((~taker_buy) & large, df['notional'], 0.0)

    grouped = df.resample('1min').agg({
        'buy_qty': 'sum',
        'sell_qty': 'sum',
        'buy_notional': 'sum',
        'sell_notional': 'sum',
        'buy_count': 'sum',
        'sell_count': 'sum',
        'large_buy_notional': 'sum',
        'large_sell_notional': 'sum',
        'notional': 'sum',
        'qty': 'sum',
    })

    res = pd.DataFrame(index=grouped.index)
    total_qty = grouped['buy_qty'] + grouped['sell_qty']
    total_notional = grouped['buy_notional'] + grouped['sell_notional']
    total_count = grouped['buy_count'] + grouped['sell_count']
    large_total = grouped['large_buy_notional'] + grouped['large_sell_notional']

    res['taker_buy_ratio'] = grouped['buy_qty'] / (total_qty + 1e-9)
    res['taker_signed_volume'] = (grouped['buy_qty'] - grouped['sell_qty']) / (total_qty + 1e-9)
    res['taker_signed_notional'] = (grouped['buy_notional'] - grouped['sell_notional']) / (total_notional + 1e-9)
    res['taker_count_imbalance'] = (grouped['buy_count'] - grouped['sell_count']) / (total_count + 1e-9)
    res['large_trade_imbalance'] = (grouped['large_buy_notional'] - grouped['large_sell_notional']) / (large_total + 1e-9)
    res['large_trade_share'] = large_total / (total_notional + 1e-9)
    res['trade_intensity'] = total_count
    res['notional_1m'] = total_notional
    res['avg_trade_size'] = total_notional / (total_count + 1e-9)
    res['taker_flow_accel'] = res['taker_signed_notional'].diff().fillna(0.0)
    res['notional_zscore_1h'] = (
        (res['notional_1m'] - res['notional_1m'].rolling(60, min_periods=15).mean())
        / (res['notional_1m'].rolling(60, min_periods=15).std() + 1e-9)
    )
    return res

def add_oi_features_v1(df_base: pd.DataFrame, df_oi: pd.DataFrame) -> pd.DataFrame:
    if df_oi.empty: return pd.DataFrame(index=df_base.index)
    oi_1m = df_oi.set_index('datetime')['openInterest'].reindex(df_base.index).ffill()
    res = pd.DataFrame(index=df_base.index)
    res['oi_change_15m'] = oi_1m.pct_change(15)
    res['oi_change_5m'] = oi_1m.pct_change(5)
    res['oi_change_1h'] = oi_1m.pct_change(60)
    res['oi_velocity'] = oi_1m.diff() / (oi_1m.shift(1) + 1e-9)
    res['oi_acceleration'] = res['oi_velocity'].diff()
    res['oi_relative_std'] = oi_1m.rolling(15).std() / (oi_1m.rolling(15).mean() + 1e-9)
    oi_ma_1h = oi_1m.rolling(60).mean()
    res['oi_momentum'] = (oi_1m - oi_ma_1h) / (oi_ma_1h + 1e-9)
    res['oi_zscore_24h'] = (oi_1m - oi_1m.rolling(1440, min_periods=240).mean()) / (oi_1m.rolling(1440, min_periods=240).std() + 1e-9)
    if 'close' in df_base.columns:
        price_change = df_base['close'].pct_change(1)
        res['oi_price_interaction'] = price_change * res['oi_velocity']
        price_change_15m = df_base['close'].pct_change(15)
        res['price_up_oi_up'] = ((price_change_15m > 0) & (res['oi_change_15m'] > 0)).astype(float)
        res['price_down_oi_up'] = ((price_change_15m < 0) & (res['oi_change_15m'] > 0)).astype(float)
        res['price_up_oi_down'] = ((price_change_15m > 0) & (res['oi_change_15m'] < 0)).astype(float)
        res['price_down_oi_down'] = ((price_change_15m < 0) & (res['oi_change_15m'] < 0)).astype(float)
        res['oi_price_divergence'] = np.sign(res['oi_change_15m']) * -np.sign(price_change_15m)
    return res

def aggregate_window_features_v1(df_full: pd.DataFrame, config: dict) -> pd.DataFrame:
    time_like_cols = set(config['features']['time_like_cols'])
    df = df_full.copy()
    df['round_label'] = df.index.floor('15min')
    all_cols      = [c for c in df.columns if c != 'round_label']
    signal_cols   = [c for c in all_cols if c not in time_like_cols]
    parts = []
    grp = df.groupby('round_label')
    sig_mean  = grp[signal_cols].mean().add_prefix('win_mean_')
    sig_std   = grp[signal_cols].std().add_prefix('win_std_')
    sig_trend = (grp[signal_cols].last() - grp[signal_cols].first()).add_prefix('win_trend_')
    parts.extend([sig_mean, sig_std, sig_trend])
    snap_all = grp[all_cols].last().add_prefix('snap_')
    parts.append(snap_all)
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
    df_aggr = pd.concat(parts, axis=1)
    for col in ['rsi_14', 'ema_dist_21', 'vwap_dev_norm', 'oi_velocity']:
        if col in df.columns:
            df['tick_idx'] = df.groupby('round_label').cumcount()
            df['win_size'] = df.groupby('round_label')['tick_idx'].transform('max')
            df['is_late']  = df['tick_idx'] > (df['win_size'] / 2)
            accel = df.groupby(['round_label', 'is_late'])[col].mean().unstack()
            if accel.shape[1] == 2:
                accel.columns = ['early', 'late']
                accel_series = (accel['late'] - accel['early']).rename(f'win_{col}_accel')
                df_aggr = pd.concat([df_aggr, accel_series], axis=1)
    df_aggr.index = df_aggr.index + pd.Timedelta(minutes=15)
    df_aggr.index.name = 'round_start'
    return df_aggr

def add_inter_round_features_v1(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    res = df.copy().sort_index()
    n_lags   = config['features']['inter_round_lags']
    roll_wins = config['features']['inter_round_rolling']
    if 'future_return' in res.columns:
        for lag in range(1, n_lags + 1):
            res[f'past_ret_lag{lag}'] = res['future_return'].shift(lag)
        res['past_ret_abs_lag1'] = res['future_return'].shift(1).abs()
    if 'target' in res.columns:
        for lag in range(1, n_lags + 1):
            res[f'past_up_lag{lag}'] = res['target'].shift(lag)
    for w in roll_wins:
        if 'future_return' in res.columns:
            past_ret = res['future_return'].shift(1)
            res[f'past_ret_sum_{w}']  = past_ret.rolling(w, min_periods=max(1, w//2)).sum()
            res[f'past_rvol_{w}']     = past_ret.rolling(w, min_periods=max(1, w//2)).std()
        if 'target' in res.columns:
            past_up  = res['target'].shift(1)
            res[f'past_up_rate_{w}']  = past_up.rolling(w, min_periods=max(1, w//2)).mean()
    if 'past_ret_lag1' in res.columns and 'past_rvol_8' in res.columns:
        res['past_ret_zscore_8'] = res['past_ret_lag1'] / (res['past_rvol_8'] + 1e-9)
    return res

def process_day_features(day, config, day_index):
    day_str = day.strftime('%Y-%m-%d')
    ob_path = PROJECT_ROOT / config['paths']['data']['orderbook'] / f"{day_str}_BTCUSDT_ob200.parquet"
    tr_path = PROJECT_ROOT / config['paths']['data']['aggtrades'] / f"BTCUSDT-aggTrades-{day_str}.parquet"
    day_micro = pd.DataFrame(index=day_index)
    if ob_path.exists():
        try:
            df_ob = pd.read_parquet(ob_path)
            df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp_ms'], unit='ms')
            df_ob = df_ob.set_index('timestamp').sort_index()
            df_ob = df_ob.resample('5s').last().dropna(subset=['bids', 'asks'])
            df_lob = compute_orderbook_features_v2(df_ob)
            df_obi = compute_obi_v1(df_ob)
            df_ofi = compute_ofi_v1(df_ob)
            micro_ob = pd.concat([df_lob, df_obi, df_ofi], axis=1).resample('1min').agg(['mean', 'std', 'last'])
            micro_ob.columns = [f"{c[0]}_{c[1]}" for c in micro_ob.columns]
            day_micro = day_micro.join(micro_ob, how='left')
        except Exception as e:
            logger.error(f"Error processing OB for {day_str}: {e}")
    if tr_path.exists():
        try:
            df_tr = pd.read_parquet(tr_path)
            df_tr['timestamp'] = pd.to_datetime(df_tr['timestamp'], unit='ms')
            df_tr = df_tr.set_index('timestamp')
            df_vpin = compute_vpin_correct(df_tr)
            df_taker = compute_taker_flow_features_v2(df_tr)
            micro_tr = df_vpin.resample('1min').agg(['mean', 'std'])
            micro_tr.columns = [f"{c[0]}_{c[1]}" for c in micro_tr.columns]
            micro_tr = micro_tr.join(df_taker, how='outer')
            day_micro = day_micro.join(micro_tr, how='left')
        except Exception as e:
            logger.error(f"Error processing Trades for {day_str}: {e}")
    return day_micro

if __name__ == "__main__":
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load 1m Klines
    klines_path = PROJECT_ROOT / config['paths']['data']['klines_1m']
    logger.info(f"Loading klines from {klines_path}...")
    df_1m = pd.read_parquet(klines_path)
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m = df_1m.set_index('timestamp')
    df_1m = df_1m.sort_index()

    # 2. Basic Features
    logger.info("Adding technical & time features...")
    df_1m = add_technical_features_v1(df_1m)
    df_1m = add_time_features_v1(df_1m)

    # 3. Open Interest Features
    oi_dir = PROJECT_ROOT / config['paths']['data']['oi'] / "BTCUSDT"
    oi_files = list(oi_dir.glob("*.parquet")) if oi_dir.exists() else []
    if oi_files:
        logger.info(f"Loading Open Interest from {oi_files[0]}...")
        df_oi = pd.read_parquet(oi_files[0])
        df_oi['datetime'] = pd.to_datetime(df_oi['datetime'])
        df_oi_feat = add_oi_features_v1(df_1m, df_oi)
        df_1m = pd.concat([df_1m, df_oi_feat], axis=1)
        logger.info("OI features added.")

    # 4. Microstructure (Parallel)
    all_days = df_1m.index.normalize().unique()
    logger.info(f"Processing microstructure for {len(all_days)} days using 10 cores...")
    tasks = []
    for day in all_days:
        day_str = day.strftime('%Y-%m-%d')
        day_index = df_1m.loc[day_str].index if day_str in df_1m.index else pd.Index([])
        tasks.append((day, config, day_index))
    micro_results = Parallel(n_jobs=10)(
        delayed(process_day_features)(d, c, idx) for d, c, idx in tqdm(tasks)
    )
    micro_parts = [res for res in micro_results if not res.empty]
    if micro_parts:
        df_micro_full = pd.concat(micro_parts).sort_index()
        df_micro_full = df_micro_full.reindex(df_1m.index)
        df_1m = pd.concat([df_1m, df_micro_full], axis=1)
        logger.info("Microstructure integrated.")

    # 5. Aggregate
    logger.info("Aggregating to 15m windows...")
    df_features = aggregate_window_features_v1(df_1m, config)
    
    # 6. Labels and Finalize
    targets_dir = PROJECT_ROOT / config['paths']['data']['labels']
    targets_path = targets_dir / "targets_v1.parquet"
    if targets_path.exists():
        logger.info(f"Loading targets from {targets_path}...")
        df_targets = pd.read_parquet(targets_path)
        df_final = df_features.join(df_targets, how='inner')
        df_final = add_inter_round_features_v1(df_final, config)
        
        output_dir = PROJECT_ROOT / config['paths']['data']['features_v1']
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "final_features.parquet"
        df_final.to_parquet(output_path)
        logger.info(f"Final features saved to {output_path} | Shape: {df_final.shape}")
    else:
        logger.error(f"Targets not found at {targets_path}. Run 02b_label_builder.py first.")
