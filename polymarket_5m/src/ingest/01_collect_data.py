import os
import pandas as pd
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_readiness(config_path: str = "/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_path = Path(config['paths']['data']['raw'])
    klines_path = Path(config['paths']['data']['klines_1m'])
    
    logger.info(f"Checking data in {raw_path}...")
    
    if klines_path.exists():
        size_mb = klines_path.stat().st_size / (1024 * 1024)
        logger.info(f"Klines 1m found: {klines_path} ({size_mb:.2f} MB)")
        
        # Quick check of date range
        df = pd.read_parquet(klines_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.set_index('open_time')
                
        start_date = df.index.min()
        end_date = df.index.max()
        logger.info(f"Data range: {start_date} to {end_date}")
        logger.info(f"Total rows: {len(df)}")
    else:
        logger.error(f"Klines 1m NOT found at {klines_path}")

    # Check aggtrades and orderbook directories
    for sub in ['aggtrades', 'orderbook']:
        p = raw_path / sub
        if p.exists():
            files = list(p.glob("*.parquet"))
            logger.info(f"{sub} directory has {len(files)} parquet files.")
        else:
            logger.warning(f"{sub} directory NOT found.")

if __name__ == "__main__":
    check_data_readiness()
