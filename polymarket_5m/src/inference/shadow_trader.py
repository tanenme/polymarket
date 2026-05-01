import pandas as pd
import numpy as np
import requests
import yaml
import time
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, UTC

# Local imports
from inference_v1 import PolymarketPredictor
import talib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShadowTrader:
    def __init__(self, config_path: str = "/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        artifacts_path = "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts"
        self.predictor = PolymarketPredictor(artifacts_path, config_path)
        
        self.shadow_log_path = Path("/run/media/rotan/New Volume/gemini3/polymarket_5m/reports/shadow_trades.csv")
        if not self.shadow_log_path.exists():
            df = pd.DataFrame(columns=['timestamp', 'p_model', 'p_market', 'decision', 'entry_price', 'target_time', 'resolved', 'is_win'])
            df.to_csv(self.shadow_log_path, index=False)

    def get_binance_klines(self, symbol="BTCUSDT", interval="1m", limit=100):
        """Fetch recent OHLCV from Binance Public API"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        res = requests.get(url)
        data = res.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def get_active_btc_token_id(self):
        """Mencari Token ID untuk pasar BTC Up/Down 15m yang sedang aktif"""
        try:
            # Query markets dengan filter 'Bitcoin'
            url = "https://clob.polymarket.com/markets"
            params = {"active": "true"}
            res = requests.get(url, params=params)
            markets = res.json()
            
            # Cari market yang mengandung 'Bitcoin' dan '15-minute' (atau format serupa)
            # Catatan: Polymarket sering menggunakan nama seperti "Bitcoin price at 5:30 PM..."
            now = datetime.now(UTC)
            target_hour = now.hour
            # Sesuaikan pencarian string berdasarkan pola nama pasar Polymarket
            for m in markets:
                desc = m.get('description', '').lower()
                if 'bitcoin' in desc and 'price' in desc:
                    # Ambil token ID untuk 'YES' (biasanya indeks 0)
                    return m['tokens'][0]['token_id'], m['description']
            return None, None
        except Exception as e:
            logger.error(f"Error finding active market: {e}")
            return None, None

    def get_polymarket_price(self):
        """Fetch current mid-price from Polymarket CLOB"""
        token_id, desc = self.get_active_btc_token_id()
        if not token_id:
            return 0.50 # Fallback
            
        try:
            url = f"https://clob.polymarket.com/book?token_id={token_id}"
            res = requests.get(url)
            book = res.json()
            
            # Mid price = (Best Bid + Best Ask) / 2
            best_bid = float(book['bids'][0]['price']) if book.get('bids') else 0.50
            best_ask = float(book['asks'][0]['price']) if book.get('asks') else 0.50
            
            mid_price = (best_bid + best_ask) / 2
            logger.info(f"Market: {desc} | Price: {mid_price:.4f}")
            return mid_price
        except Exception as e:
            logger.error(f"Error fetching CLOB price: {e}")
            return 0.50

    def compute_realtime_features(self, df_1m):
        """Simplified version of 02_feature_engineering for a single snapshot"""
        df = df_1m.copy()
        close = df['close'].values
        
        # Technicals
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        macd, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_hist'] = macd_hist
        
        # Realized Vol
        log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
        df['rvol_300'] = log_ret.rolling(5).std() * np.sqrt(5)
        
        latest_features = pd.DataFrame(index=[df.index[-1]])
        for col in self.predictor.selected_features:
            if col in df.columns:
                latest_features[col] = df[col].iloc[-1]
            elif col.startswith('snap_') and col[5:] in df.columns:
                latest_features[col] = df[col[5:]].iloc[-1]
            else:
                latest_features[col] = 0.0 # Placeholder
                
        return latest_features

    def run_one_iteration(self):
        logger.info("Starting Shadow Trade iteration...")
        
        try:
            df_1m = self.get_binance_klines()
            current_price = df_1m['close'].iloc[-1]
            p_market = self.get_polymarket_price()
            
            X = self.compute_realtime_features(df_1m)
            
            p_model = self.predictor.predict_probability(X)[0]
            decision = self.predictor.get_trade_decision(p_model, p_market)
            
            now_utc = datetime.now(UTC)
            target_time = (now_utc + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
            
            new_trade = {
                'timestamp': now_utc.strftime('%Y-%m-%d %H:%M:%S'),
                'p_model': p_model,
                'p_market': p_market,
                'decision': decision['decision'],
                'entry_price': current_price,
                'target_time': target_time,
                'resolved': False,
                'is_win': None
            }
            
            df_log = pd.read_csv(self.shadow_log_path)
            df_log = pd.concat([df_log, pd.DataFrame([new_trade])], ignore_index=True)
            df_log.to_csv(self.shadow_log_path, index=False)
            
            logger.info(f"Decision: {decision['decision']} | Model: {p_model:.4f} | Market: {p_market:.2f}")
            
        except Exception as e:
            logger.error(f"Error in shadow trader iteration: {e}")

    def resolve_old_trades(self):
        """Check trades from 15 mins ago against current price"""
        if not self.shadow_log_path.exists(): return
        
        df_log = pd.read_csv(self.shadow_log_path)
        if df_log.empty: return
        
        try:
            df_1m = self.get_binance_klines()
            current_price = df_1m['close'].iloc[-1]
            now = datetime.now(UTC)
            
            updated = False
            for idx, row in df_log.iterrows():
                if not row['resolved'] and row['decision'] != 'SKIP':
                    target_dt = datetime.strptime(row['target_time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                    if now >= target_dt:
                        actual_up = current_price > row['entry_price']
                        is_win = (row['decision'] == 'BET_YES' and actual_up) or \
                                 (row['decision'] == 'BET_NO' and not actual_up)
                        
                        df_log.at[idx, 'resolved'] = True
                        df_log.at[idx, 'is_win'] = is_win
                        updated = True
                        logger.info(f"Resolved trade from {row['timestamp']}: {'WIN' if is_win else 'LOSS'} (Price: {row['entry_price']} -> {current_price})")
            
            if updated:
                df_log.to_csv(self.shadow_log_path, index=False)
        except Exception as e:
            logger.error(f"Error resolving trades: {e}")

if __name__ == "__main__":
    trader = ShadowTrader()
    logger.info("Shadow Trader active (Modernized UTC). Press Ctrl+C to stop.")
    
    while True:
        trader.resolve_old_trades()
        
        now = datetime.now(UTC)
        if now.minute % 15 == 0 and now.second < 10:
            trader.run_one_iteration()
            time.sleep(15)
            
        time.sleep(5)
