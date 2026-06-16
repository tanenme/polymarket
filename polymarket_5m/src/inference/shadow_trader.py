import pandas as pd
import numpy as np
import requests
import yaml
import time
import logging
import json
import os
import talib
import sys
import threading
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta, UTC

# Local imports
from inference_v1 import PolymarketPredictor
from find_markets import PolymarketMarketFinder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically find project root (assuming script is in src/inference/shadow_trader.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import feature engineering module (filename starts with digit -> importlib).
# CRITICAL: live features must be IDENTICAL to training, so we reuse the exact
# same functions instead of reimplementing them here.
_fe_path = PROJECT_ROOT / "src/features/02_feature_engineering.py"
_fe_spec = importlib.util.spec_from_file_location("feature_engineering_v1", _fe_path)
fe = importlib.util.module_from_spec(_fe_spec)
_fe_spec.loader.exec_module(fe)

class ShadowTrader:
    def __init__(self, config_path: str = None, test_mode: bool = False, forced_threshold: float = None):
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.test_mode = test_mode
        self.forced_threshold = forced_threshold
        
        # Resolve artifacts path
        artifacts_rel = self.config.get('paths', {}).get('models', {}).get('artifacts', "models/artifacts")
        artifacts_path = str(PROJECT_ROOT / artifacts_rel)
        
        self.predictor = PolymarketPredictor(artifacts_path, config_path)
        
        # Resolve reports path
        reports_rel = self.config.get('paths', {}).get('reports', "reports")
        reports_path = PROJECT_ROOT / reports_rel
        reports_path.mkdir(parents=True, exist_ok=True)
        
        # Suffix for files based on threshold and test mode
        suffix = ""
        if self.forced_threshold is not None:
            suffix += f"_t{self.forced_threshold}"
        if self.test_mode:
            suffix += "_test"
            
        self.shadow_log_path = reports_path / f"shadow_trades{suffix}.csv"
        self.bankroll_path = reports_path / f"shadow_bankroll{suffix}.json"
        
        self.buffer_1m = pd.DataFrame()
        self.round_history = pd.DataFrame()
        
        # Buffer for real-time Bybit Orderbook snapshots
        self.ob_collector_buffer = [] 
        self.ob_lock = threading.Lock()
        
        # Start Background OB Collector (200ms interval to match training data)
        self.stop_threads = False
        self.ob_thread = threading.Thread(target=self._ob_collector_loop, daemon=True)
        self.ob_thread.start()
        
        # Define Columns
        self.log_cols = ['timestamp', 'p_model', 'p_market', 'decision', 'bet_amount', 'execution_price', 'entry_price_exchange', 'target_time', 'resolved', 'is_win', 'net_profit', 'bankroll', 'win_return', 'win_rate', 'max_drawdown']
        
        # Initialize log if not exists or empty
        if not self.shadow_log_path.exists() or self.shadow_log_path.stat().st_size == 0:
            df = pd.DataFrame(columns=self.log_cols)
            df.to_csv(self.shadow_log_path, index=False)
            logger.info(f"Initialized new shadow log at {self.shadow_log_path}")

        # Initialize Bankroll
        self.initial_bankroll = self.config['trade_decision'].get('initial_bankroll', 70.0)
        self.bankroll = self.load_bankroll()
        
        # Load history
        if self.shadow_log_path.exists():
            try:
                hist = pd.read_csv(self.shadow_log_path)
                if not hist.empty:
                    hist['round_start'] = pd.to_datetime(hist['timestamp'])
                    self.round_history = hist.set_index('round_start')
            except Exception as e:
                logger.warning(f"Could not load trade history: {e}")

    def load_bankroll(self):
        if self.bankroll_path.exists():
            try:
                with open(self.bankroll_path, 'r') as f:
                    return json.load(f)['bankroll']
            except: pass
        return self.initial_bankroll

    def save_bankroll(self):
        with open(self.bankroll_path, 'w') as f:
            json.dump({'bankroll': self.bankroll, 'last_update': str(datetime.now(UTC))}, f)

    def add_technical_features_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        close, high, low, volume = res['close'].values, res['high'].values, res['low'].values, res['volume'].values
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
            res[f'rvol_{window}'] = pd.Series(log_ret).rolling(n_candles).std().values * np.sqrt(n_candles)
        return res

    def add_time_features_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df.index
        hours = ts.hour + ts.minute / 60.0
        dows  = ts.dayofweek
        df['sin_hour'], df['cos_hour'] = np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)
        df['sin_dow'],  df['cos_dow']  = np.sin(2*np.pi*dows/7),  np.cos(2*np.pi*dows/7)
        slot_15m = ts.hour * 4 + ts.minute // 15
        df['sin_15m'] = np.sin(2 * np.pi * slot_15m / 96)
        df['cos_15m'] = np.cos(2 * np.pi * slot_15m / 96)
        df['is_us_session']     = ((ts.hour >= 13) & (ts.hour < 21)).astype(float)
        df['is_london_session'] = ((ts.hour >= 7)  & (ts.hour < 16)).astype(float)
        df['is_overlap']        = df['is_us_session'] * df['is_london_session']
        df['is_weekend']        = (dows >= 5).astype(float)
        df['is_roll_hour']       = (ts.hour % 8 == 0).astype(float)
        df['minutes_since_roll'] = ((ts.hour % 8) * 60 + ts.minute).astype(float)
        return df

    def add_oi_features_v1(self, df_base: pd.DataFrame, df_oi: pd.DataFrame) -> pd.DataFrame:
        if df_oi.empty: return pd.DataFrame(index=df_base.index)
        # Match training: reindex 5m OI to 1m base and ffill
        oi_1m = df_oi.set_index('timestamp')['openInterest'].reindex(df_base.index).ffill()
        res = pd.DataFrame(index=df_base.index)
        res['oi_change_15m'] = oi_1m.pct_change(15)
        res['oi_velocity'] = oi_1m.diff() / (oi_1m.shift(1) + 1e-9)
        res['oi_relative_std'] = oi_1m.rolling(15).std() / (oi_1m.rolling(15).mean() + 1e-9)
        oi_ma_1h = oi_1m.rolling(60).mean()
        res['oi_momentum'] = (oi_1m - oi_ma_1h) / (oi_ma_1h + 1e-9)
        if 'close' in df_base.columns:
            price_change = df_base['close'].pct_change(1)
            res['oi_price_interaction'] = price_change * res['oi_velocity']
        return res

    def aggregate_window_features_v1(self, df_full: pd.DataFrame) -> pd.DataFrame:
        config = self.config
        time_like_cols = set(config['features']['time_like_cols'])
        df = df_full.copy()
        df['round_label'] = df.index.floor('15min')
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'round_label']
        all_cols      = [c for c in df.columns if c != 'round_label']
        signal_cols   = [c for c in numeric_cols if c not in time_like_cols]
        parts = []
        grp = df.groupby('round_label')
        if signal_cols:
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

    def add_inter_round_features_v1(self, df_aggr: pd.DataFrame) -> pd.DataFrame:
        res = df_aggr.copy()
        n_lags = self.config['features']['inter_round_lags']
        roll_wins = self.config['features']['inter_round_rolling']
        for lag in range(1, n_lags + 1):
            res[f'past_ret_lag{lag}'] = np.nan
            res[f'past_up_lag{lag}'] = np.nan
        res['past_ret_abs_lag1'] = np.nan
        for w in roll_wins:
            res[f'past_ret_sum_{w}'] = np.nan
            res[f'past_rvol_{w}'] = np.nan
            res[f'past_up_rate_{w}'] = np.nan
        res['past_ret_zscore_8'] = np.nan
        if self.round_history.empty:
            return res
        combined = pd.concat([self.round_history, res]).sort_index()
        if 'win_return' in combined.columns:
            for lag in range(1, n_lags + 1):
                res[f'past_ret_lag{lag}'] = combined['win_return'].shift(lag).iloc[-1]
            res['past_ret_abs_lag1'] = combined['win_return'].shift(1).abs().iloc[-1]
        if 'target' in combined.columns:
            for lag in range(1, n_lags + 1):
                res[f'past_up_lag{lag}'] = combined['target'].shift(lag).iloc[-1]
        for w in roll_wins:
            if 'win_return' in combined.columns:
                past_ret = combined['win_return'].shift(1)
                res[f'past_ret_sum_{w}'] = past_ret.rolling(w, min_periods=max(1, w//2)).sum().iloc[-1]
                res[f'past_rvol_{w}'] = past_ret.rolling(w, min_periods=max(1, w//2)).std().iloc[-1]
            if 'target' in combined.columns:
                past_up = combined['target'].shift(1)
                res[f'past_up_rate_{w}'] = past_up.rolling(w, min_periods=max(1, w//2)).mean().iloc[-1]
        if 'past_ret_lag1' in res.columns and 'past_rvol_8' in res.columns:
            res['past_ret_zscore_8'] = res['past_ret_lag1'] / (res['past_rvol_8'] + 1e-9)
        return res

    def _ob_collector_loop(self):
        """Background thread loop for high-frequency OB collection."""
        logger.info("Starting background Bybit Orderbook collector (200ms)...")
        while not self.stop_threads:
            start_t = time.time()
            self.collect_bybit_orderbook()
            elapsed = time.time() - start_t
            sleep_time = max(0.01, 0.2 - elapsed) # Target 200ms
            time.sleep(sleep_time)

    def collect_bybit_orderbook(self):
        """Fetches L50 Bybit Orderbook and stores in memory buffer.

        Stores raw bids/asks as JSON strings (same schema as the training
        orderbook parquet) so we can reuse the exact training feature funcs.
        """
        url = "https://api.bybit.com/v5/market/orderbook"
        params = {"category": "spot", "symbol": "BTCUSDT", "limit": 50}
        try:
            resp = requests.get(url, params=params, timeout=2).json()
            if resp.get('retCode') == 0:
                res = resp['result']
                # Bybit returns [["price","qty"], ...]; training parquet stores
                # the same nested-list JSON in 'bids'/'asks' columns.
                snapshot = {
                    'timestamp_ms': int(res['ts']),
                    'bids': json.dumps(res.get('b', [])),
                    'asks': json.dumps(res.get('a', [])),
                }
                with self.ob_lock:
                    self.ob_collector_buffer.append(snapshot)
                    # Keep ~20 minutes (200ms cadence -> ~6000 snapshots)
                    if len(self.ob_collector_buffer) > 7000:
                        self.ob_collector_buffer.pop(0)
        except Exception:
            pass

    def sync_data(self) -> bool:
        logger.info("Syncing historical data from BYBIT for 100% consistency...")
        # 1. Fetch Klines (Spot)
        klines_url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "spot", "symbol": "BTCUSDT", "interval": "1", "limit": 300}
        try:
            resp = requests.get(klines_url, params=params, timeout=10).json()
            if resp.get('retCode') == 0:
                klines_list = resp['result']['list']
                df = pd.DataFrame(klines_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df = df.sort_index()
            else:
                logger.error(f"Failed to fetch klines from Bybit: {resp.get('retMsg')}")
                return False
        except Exception as e:
            logger.error(f"Bybit API Error (Klines): {e}")
            return False

        # 2. Fetch Open Interest (Linear/Perp)
        oi_url = "https://api.bybit.com/v5/market/open-interest"
        params = {"category": "linear", "symbol": "BTCUSDT", "intervalTime": "5min", "limit": 200}
        try:
            resp = requests.get(oi_url, params=params, timeout=10).json()
            if resp.get('retCode') == 0:
                oi_list = resp['result']['list']
                oi_df = pd.DataFrame(oi_list)
                # Training's add_oi_features_v1 expects a 'datetime' column
                oi_df['datetime'] = pd.to_datetime(oi_df['timestamp'].astype(int), unit='ms')
                oi_df['openInterest'] = oi_df['openInterest'].astype(float)

                # Reuse training OI feature func for full parity (oi_momentum,
                # oi_change_5m/15m/1h, oi_acceleration, oi_zscore_24h, etc.)
                oi_feats = fe.add_oi_features_v1(df, oi_df)
                df = pd.concat([df, oi_feats], axis=1).ffill().fillna(0)
            else:
                logger.warning(f"Failed to fetch OI from Bybit: {resp.get('retMsg')}")
        except Exception as e:
            logger.warning(f"Bybit API Error (OI): {e}")

        # 3. Fetch Recent Trades (Spot) -> VPIN + Taker Flow (reuse training funcs)
        trades_url = "https://api.bybit.com/v5/market/recent-trade"
        params = {"category": "spot", "symbol": "BTCUSDT", "limit": 1000}
        try:
            resp = requests.get(trades_url, params=params, timeout=10).json()
            if resp.get('retCode') == 0:
                trades_list = resp['result']['list']
                tr_df = pd.DataFrame(trades_list)
                # Build the exact schema compute_vpin/taker expect:
                # index=datetime, cols: qty, price, is_buyer_maker
                tr_df.index = pd.to_datetime(tr_df['time'].astype(int), unit='ms')
                tr_df['qty'] = tr_df['size'].astype(float)
                tr_df['price'] = tr_df['price'].astype(float)
                tr_df['is_buyer_maker'] = tr_df['side'] == 'Sell'
                tr_df = tr_df.sort_index()

                # VPIN (training: resample 1min ['mean','std'])
                df_vpin = fe.compute_vpin_correct(tr_df)
                df_vpin.index = tr_df.index
                micro_tr = df_vpin.resample('1min').agg(['mean', 'std'])
                micro_tr.columns = [f"{c[0]}_{c[1]}" for c in micro_tr.columns]

                # Taker flow features (training: already 1min)
                df_taker = fe.compute_taker_flow_features_v2(tr_df)
                micro_tr = micro_tr.join(df_taker, how='outer')

                df = df.join(micro_tr, how='left').fillna(0)
            else:
                logger.warning(f"Failed to fetch trades from Bybit: {resp.get('retMsg')}")
        except Exception as e:
            logger.warning(f"Bybit API Error (Trades): {e}")

        # 4. Process Bybit L50 Orderbook Buffer -> full LOB + OBI + OFI
        #    (reuse the SAME training functions for 100% feature parity)
        with self.ob_lock:
            local_ob_buffer = list(self.ob_collector_buffer)

        if local_ob_buffer:
            try:
                ob_df = pd.DataFrame(local_ob_buffer)
                ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp_ms'], unit='ms')
                ob_df = ob_df.set_index('timestamp').sort_index()
                # Training resamples OB to 5s before computing features
                ob_df = ob_df.resample('5s').last().dropna(subset=['bids', 'asks'])

                df_lob = fe.compute_orderbook_features_v2(ob_df)
                df_obi = fe.compute_obi_v1(ob_df)
                df_ofi = fe.compute_ofi_v1(ob_df)
                micro_ob = pd.concat([df_lob, df_obi, df_ofi], axis=1).resample('1min').agg(['mean', 'std', 'last'])
                micro_ob.columns = [f"{c[0]}_{c[1]}" for c in micro_ob.columns]

                df = df.join(micro_ob, how='left').ffill().fillna(0)
            except Exception as e:
                logger.warning(f"Orderbook feature error: {e}")

        self.buffer_1m = df
        return True

    def get_polymarket_data(self):
        """Discovers active 15m BTC market using the centralized Finder."""
        market = PolymarketMarketFinder.find_bitcoin_price_market()
        if market:
            return market['best_bid'], market['best_ask'], (market['best_bid'] + market['best_ask'])/2, market['question']
        return 0.0, 0.0, 0.50, "N/A"

    def get_performance_stats(self, df=None):
        if df is None:
            if not self.shadow_log_path.exists(): return 0.0, 0.0
            try:
                df = pd.read_csv(self.shadow_log_path)
            except: return 0.0, 0.0
            
        try:
            # Type Fixes to prevent errors
            df['resolved'] = df['resolved'].astype(object)
            if 'is_win' in df.columns: df['is_win'] = df['is_win'].astype(object)
            for col in ['bankroll', 'win_rate', 'max_drawdown']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                
            resolved = df[df['resolved'] == True].copy()
            if resolved.empty: return 0.0, 0.0
                
            win_rate = resolved['is_win'].astype(bool).mean()
            bankroll_history = resolved['bankroll'].astype(float).values
            full_history = np.insert(bankroll_history, 0, self.initial_bankroll)
            peaks = np.maximum.accumulate(full_history)
            drawdowns = (peaks - full_history) / (peaks + 1e-9)
            max_dd = np.max(drawdowns)
            return win_rate, max_dd
        except: return 0.0, 0.0

    def run_prediction(self):
        logger.info(f"\n" + "="*60)
        logger.info(f"RUNNING PREDICTION ({'TEST' if self.test_mode else 'NORMAL'} MODE)")
        logger.info("="*60)

        # GUARD: pastikan buffer OB sudah cukup terisi sebelum prediksi.
        # Window fitur = 15 menit. OB @200ms idealnya ~4500 snapshot/15m.
        # Minimal 12 menit data (~3600 snapshot) agar fitur OB stabil & valid.
        MIN_OB_SNAPSHOTS = 3600
        with self.ob_lock:
            ob_count = len(self.ob_collector_buffer)
        if ob_count < MIN_OB_SNAPSHOTS:
            mins = ob_count * 0.2 / 60.0
            logger.warning(
                f"Status: SKIP (buffer OB belum siap: {ob_count}/{MIN_OB_SNAPSHOTS} snapshot, ~{mins:.1f} menit). "
                f"Tunggu collector mengisi window 15 menit sebelum taruhan."
            )
            return

        # 1. Sync data Bybit
        if not self.sync_data():
            logger.error("Status: ABORTED (Could not sync data from Bybit).")
            return

        if self.buffer_1m.empty:
            logger.error("Status: ABORTED (Buffer is empty after sync).")
            return
        
        # 2. Get Polymarket Data
        best_bid, best_ask, p_market_mid, desc = self.get_polymarket_data()
        spread = best_ask - best_bid
        
        logger.info(f"Market: {desc}")
        logger.info(f"Price : Bid {best_bid:.4f} | Ask {best_ask:.4f} | Mid {p_market_mid:.4f}")
        logger.info(f"Spread: {spread:.4f} ({spread*100:.2f}%)")
        
        # --- SAFEGUARDS ---
        if best_bid <= 0.0 or best_ask >= 1.0 or best_ask <= best_bid:
            logger.warning("Status: REJECTED (Invalid market data / No Liquidity). Skipping.")
            return
            
        if spread > 0.05:
            logger.warning(f"Status: REJECTED (Spread {spread:.4f} > 0.05). Skipping for safety.")
            return
        # ------------------

        # 3. Feature Engineering & Prediction
        # Use the EXACT training functions for technical/time/aggregate so the
        # live feature columns match the trained model 1:1. Inter-round features
        # stay local because they depend on live round_history.
        df_1m = fe.add_technical_features_v1(self.buffer_1m)
        df_1m = fe.add_time_features_v1(df_1m)
        df_aggr = fe.aggregate_window_features_v1(df_1m.tail(15), self.config)
        df_final = self.add_inter_round_features_v1(df_aggr)

        # GUARD: jangan taruhan kalau fitur belum 100% siap.
        # Prediksi dgn fitur di-fill 0 = sinyal tidak valid -> bisa rugi.
        missing = [f for f in self.predictor.selected_features if f not in df_final.columns]
        if missing:
            logger.warning(
                f"Status: SKIP (fitur belum siap: {len(missing)}/{len(self.predictor.selected_features)} hilang). "
                f"Buffer OB mungkin belum penuh. Contoh: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
            return

        X = df_final[self.predictor.selected_features]

        # GUARD: cek tidak ada NaN/inf di fitur (data live belum matang).
        # Kolom inter-round (past_*) memang NaN di awal sesi -> diperbolehkan
        # (model di-train dgn invalidasi serupa). Sisanya WAJIB valid.
        non_lag = [f for f in self.predictor.selected_features if not f.startswith('past_')]
        bad = X[non_lag].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).iloc[-1]
        if bad:
            nan_cols = X[non_lag].columns[X[non_lag].replace([np.inf, -np.inf], np.nan).isna().iloc[-1]].tolist()
            logger.warning(
                f"Status: SKIP (fitur mengandung NaN/inf, data live belum matang). "
                f"Contoh: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}"
            )
            return

        p_model = self.predictor.predict_probability(X)[0]
        
        # --- MANUAL THRESHOLD FILTER ---
        confidence = max(p_model, 1 - p_model)
        if self.forced_threshold and confidence < self.forced_threshold:
            logger.info(f"Status: SKIP (Confidence {confidence:.4f} < Manual Threshold {self.forced_threshold})")
            return
        # -------------------------------

        # 4. Decision Logic (EV & Sizing)
        decision = self.predictor.get_trade_decision(p_model, best_bid, best_ask)
        
        logger.info(f"Model Probability (UP): {p_model:.4f}")
        logger.info(f"Edge/EV Real          : {decision['ev']:.4f}")
        
        if decision['decision'] == 'SKIP':
            logger.info("Decision: SKIP (EV does not meet threshold)")
            return
            
        execution_price = decision['execution_price']
        bet_amount = self.bankroll * decision['bet_size']
        
        logger.info(f"Decision: {decision['decision']}")
        logger.info(f"Execution Price: {execution_price:.2f}")
        logger.info(f"Bet Size (Kelly): {decision['bet_size']:.2%} (${bet_amount:.2f})")
        
        # 5. Logging & History
        now_utc = datetime.now(UTC)
        target_mins = 2 if self.test_mode else 15
        target_time = (now_utc + timedelta(minutes=target_mins)).strftime('%Y-%m-%d %H:%M:%S')
        
        wr, mdd = self.get_performance_stats()
        new_trade = {
            'timestamp': now_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'p_model': p_model, 
            'p_market': p_market_mid,
            'decision': decision['decision'],
            'bet_amount': bet_amount, 
            'execution_price': execution_price,
            'entry_price_exchange': self.buffer_1m['close'].iloc[-1],
            'target_time': target_time, 'resolved': False, 'is_win': None, 'net_profit': 0.0,
            'bankroll': self.bankroll, 'win_return': df_aggr['win_return'].iloc[-1],
            'win_rate': wr, 'max_drawdown': mdd
        }
        
        # Append to log file
        df_log = pd.read_csv(self.shadow_log_path) if self.shadow_log_path.exists() else pd.DataFrame()
        df_log = pd.concat([df_log, pd.DataFrame([new_trade])], ignore_index=True)
        df_log.to_csv(self.shadow_log_path, index=False)
        
        # Update round history
        new_hist = pd.DataFrame({'target': [np.nan], 'win_return': [df_aggr['win_return'].iloc[-1]]}, index=[pd.to_datetime(new_trade['timestamp'])])
        self.round_history = pd.concat([self.round_history, new_hist])
        
        logger.info("Trade recorded in shadow log.")
        logger.info("="*60)

    def resolve_trades(self):
        if not self.shadow_log_path.exists(): return
        try:
            df_log = pd.read_csv(self.shadow_log_path)
            if df_log.empty: return
        except: return

        # IMPORTANT: Fix dtypes immediately after read to prevent VPS crash
        df_log['resolved'] = df_log['resolved'].astype(object)
        if 'is_win' in df_log.columns: df_log['is_win'] = df_log['is_win'].astype(object)
        for c in ['bankroll', 'net_profit', 'win_rate', 'max_drawdown', 'bet_amount']:
            if c in df_log.columns: df_log[c] = pd.to_numeric(df_log[c], errors='coerce').astype(float)
        
        now = datetime.now(UTC)
        try:
            # Bybit V5 Tickers
            ticker_url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "spot", "symbol": "BTCUSDT"}
            resp = requests.get(ticker_url, params=params, timeout=10).json()
            if resp.get('retCode') == 0:
                curr_p = float(resp['result']['list'][0]['lastPrice'])
            else:
                return
        except: return
        updated = False
        for idx, row in df_log.iterrows():
            if row['resolved'] != True and row['decision'] != 'SKIP':
                target_dt = datetime.strptime(row['target_time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                if now >= target_dt:
                    is_win = (row['decision'] == 'BET_YES' and curr_p > row['entry_price_exchange']) or \
                             (row['decision'] == 'BET_NO' and curr_p <= row['entry_price_exchange'])

                    exec_p = row['execution_price']
                    shares = row['bet_amount'] / (exec_p + 1e-9)
                    if is_win:
                        gross_payoff = shares * 1.0
                        profit = gross_payoff - row['bet_amount']
                        net_profit = profit * (1 - self.config['trade_decision']['polymarket_fee'])
                    else:
                        net_profit = -row['bet_amount']
                    self.bankroll += net_profit
                    df_log.at[idx, 'resolved'] = True
                    df_log.at[idx, 'is_win'] = bool(is_win)
                    df_log.at[idx, 'net_profit'] = float(net_profit)
                    df_log.at[idx, 'bankroll'] = float(self.bankroll)
                    
                    # Update stats after this resolution
                    wr, mdd = self.get_performance_stats(df_log)
                    df_log.at[idx, 'win_rate'] = float(wr)
                    df_log.at[idx, 'max_drawdown'] = float(mdd)
                    
                    updated = True
                    ts = pd.to_datetime(row['timestamp'])
                    if ts in self.round_history.index:
                        self.round_history.at[ts, 'target'] = 1 if is_win else 0
                    logger.info(f"RESOLVED: {'WIN' if is_win else 'LOSS'} | Net: ${net_profit:+.2f} | Balance: ${self.bankroll:.2f}")
        if updated: 
            df_log.to_csv(self.shadow_log_path, index=False)
            self.save_bankroll()
            wr, mdd = self.get_performance_stats(df_log)
            logger.info(f"PERFORMANCE SUMMARY: WinRate: {wr:.2%} | MaxDD: {mdd:.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shadow Trader for Polymarket BTC 15m")
    parser.add_argument("--test", action="store_true", help="Run in test mode (2m rounds)")
    parser.add_argument("--threshold", "--t", type=float, help="Manual probability threshold (e.g. 0.585)")
    args = parser.parse_args()
    
    trader = ShadowTrader(test_mode=args.test, forced_threshold=args.threshold)
    
    logger.info(f"Shadow Trader Mode: {'TEST (2m)' if args.test else 'NORMAL (15m)'}")
    if args.threshold:
        logger.info(f"Manual Threshold Applied: {args.threshold}")
    
    logger.info(f"Current Bankroll: ${trader.bankroll:.2f}")
    
    while True:
        trader.resolve_trades()
        
        now = datetime.now(UTC)
        interval = 2 if args.test else 15
        if now.minute % interval == 0 and now.second < 10:
            trader.run_prediction()
            time.sleep(20)
        time.sleep(1) # Check resolution every second
