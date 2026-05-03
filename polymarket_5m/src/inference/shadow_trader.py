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

class ShadowTrader:
    def __init__(self, config_path: str = None, test_mode: bool = False):
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.test_mode = test_mode
        
        # Resolve artifacts path
        artifacts_rel = self.config.get('paths', {}).get('models', {}).get('artifacts', "models/artifacts")
        artifacts_path = str(PROJECT_ROOT / artifacts_rel)
        
        self.predictor = PolymarketPredictor(artifacts_path, config_path)
        
        # Resolve reports path
        reports_rel = self.config.get('paths', {}).get('reports', "reports")
        reports_path = PROJECT_ROOT / reports_rel
        reports_path.mkdir(parents=True, exist_ok=True)
        
        self.shadow_log_path = reports_path / "shadow_trades.csv"
        self.bankroll_path = reports_path / "shadow_bankroll.json"
        
        self.buffer_1m = pd.DataFrame()
        self.round_history = pd.DataFrame()
        
        # Define Columns
        self.log_cols = ['timestamp', 'p_model', 'p_market', 'decision', 'bet_amount', 'execution_price', 'entry_price_binance', 'target_time', 'resolved', 'is_win', 'net_profit', 'bankroll', 'win_return', 'win_rate', 'max_drawdown']
        
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

    def sync_data(self):
        logger.info("Syncing historical data for 100% consistency...")
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=300"
        df = pd.DataFrame(requests.get(url).json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
        tr_url = "https://api.binance.com/api/v3/aggTrades?symbol=BTCUSDT&limit=5000"
        tr_df = pd.DataFrame(requests.get(tr_url).json())
        tr_df['T'] = pd.to_datetime(tr_df['T'], unit='ms')
        tr_df['q'] = tr_df['q'].astype(float)
        tr_df['buy_vol'] = np.where(tr_df['m'] == False, tr_df['q'], 0)
        tr_df['sell_vol'] = np.where(tr_df['m'] == True, tr_df['q'], 0)
        vpin_res = tr_df.set_index('T').resample('1min').apply(lambda x: abs(x['buy_vol'].sum() - x['sell_vol'].sum()) / (x['q'].sum() + 1e-9))
        vpin_1m = pd.DataFrame(index=vpin_res.index)
        vpin_1m['vpin_mean'] = vpin_res.values
        vpin_1m['vpin_std'] = 0.0
        df = df.join(vpin_1m, how='left').fillna(0)
        df['obi_mean'], df['obi_std'] = 0.0, 0.0 
        df['ofi_mean'], df['ofi_std'] = 0.0, 0.0
        self.buffer_1m = df

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
        logger.info(f"Running Prediction ({'TEST' if self.test_mode else 'NORMAL'} MODE)...")
        self.sync_data() 
        best_bid, best_ask, p_market_mid, desc = self.get_polymarket_data()
        if best_bid == 0.0 or best_ask == 0.0:
            logger.warning("Invalid market data (0.0). Skipping prediction.")
            return
        obi = (best_bid - (1 - best_ask))
        self.buffer_1m.iloc[-1, self.buffer_1m.columns.get_loc('obi_mean')] = obi
        df_1m = self.add_technical_features_v1(self.buffer_1m)
        df_1m = self.add_time_features_v1(df_1m)
        df_aggr = self.aggregate_window_features_v1(df_1m.tail(15))
        df_final = self.add_inter_round_features_v1(df_aggr)
        X = df_final[self.predictor.selected_features]
        p_model = self.predictor.predict_probability(X)[0]
        decision = self.predictor.get_trade_decision(p_model, p_market_mid)
        execution_price = 0.0
        if decision['decision'] == 'BET_YES':
            execution_price = best_ask
        elif decision['decision'] == 'BET_NO':
            execution_price = 1 - best_bid
        if decision['decision'] == 'SKIP':
            logger.info(f"Model: {p_model:.4f} | Market: {p_market_mid:.4f} | Decision: SKIP")
            return
        now_utc = datetime.now(UTC)
        target_mins = 2 if self.test_mode else 15
        target_time = (now_utc + timedelta(minutes=target_mins)).strftime('%Y-%m-%d %H:%M:%S')
        bet_amount = self.bankroll * decision['bet_size']
        wr, mdd = self.get_performance_stats()
        new_trade = {
            'timestamp': now_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'p_model': p_model, 'p_market': p_market_mid, 'decision': decision['decision'],
            'bet_amount': bet_amount, 'execution_price': execution_price,
            'entry_price_binance': self.buffer_1m['close'].iloc[-1],
            'target_time': target_time, 'resolved': False, 'is_win': None, 'net_profit': 0.0,
            'bankroll': self.bankroll, 'win_return': df_aggr['win_return'].iloc[-1],
            'win_rate': wr, 'max_drawdown': mdd
        }
        df_log = pd.read_csv(self.shadow_log_path) if self.shadow_log_path.exists() else pd.DataFrame()
        df_log = pd.concat([df_log, pd.DataFrame([new_trade])], ignore_index=True)
        df_log.to_csv(self.shadow_log_path, index=False)
        new_hist = pd.DataFrame({'target': [np.nan], 'win_return': [df_aggr['win_return'].iloc[-1]]}, index=[pd.to_datetime(new_trade['timestamp'])])
        self.round_history = pd.concat([self.round_history, new_hist])
        logger.info(f"MARKET: {desc[:50]}... | Decision: {decision['decision']} | Price: {execution_price:.4f}")

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
            curr_p = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT").json()['price'])
        except: return
        updated = False
        for idx, row in df_log.iterrows():
            if row['resolved'] != True and row['decision'] != 'SKIP':
                target_dt = datetime.strptime(row['target_time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                if now >= target_dt:
                    is_win = (row['decision'] == 'BET_YES' and curr_p > row['entry_price_binance']) or \
                             (row['decision'] == 'BET_NO' and curr_p <= row['entry_price_binance'])
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
    is_test = "--test" in sys.argv
    trader = ShadowTrader(test_mode=is_test)
    logger.info(f"Shadow Trader Mode: {'TEST (2m)' if is_test else 'NORMAL (15m)'}")
    logger.info(f"Current Bankroll: ${trader.bankroll:.2f}")
    while True:
        trader.resolve_trades()
        now = datetime.now(UTC)
        interval = 2 if is_test else 15
        if now.minute % interval == 0 and now.second < 10:
            trader.run_prediction()
            time.sleep(20)
        time.sleep(5)
