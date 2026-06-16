#!/usr/bin/env python3
"""
Script 01: Data Collection untuk Polymarket BTC 15-Minute Prediction Model

Fungsi:
1. Download data historis OHLCV 1m dan 15m dari Binance Data Vision
2. Download funding rate, open interest, dan long/short ratio
3. Collect data real-time dari Binance REST API
4. Validasi dan simpan data ke format parquet

BATASAN:
- JANGAN simpan data lebih dari 10 detik per snapshot OB (RAM constraint)
- JANGAN query REST API lebih dari 1 kali per detik (rate limit)
- SELALU simpan timestamp dalam UTC milliseconds
- SELALU validasi data sebelum simpan: bid < ask, qty > 0
"""

import os
import sys
import yaml
import time
import logging
import argparse
import zipfile
import io
import gc
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/run/media/rotan/New Volume/claude/polymarket_15m/logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """Kelas untuk mengumpulkan data dari Binance"""
    
    def __init__(self, config_path: str = "/run/media/rotan/New Volume/gemini3/orderbook-kline-aggtrade/config.yaml"):
        """Inisialisasi collector dengan konfigurasi"""
        self.config = self.load_config(config_path)
        self.setup_paths()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def load_config(self, config_path: str) -> Dict:
        """Load konfigurasi dari file YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Konfigurasi loaded dari {config_path}")
            return config
        except Exception as e:
            logger.error(f"Gagal load konfigurasi: {e}")
            raise
    
    def setup_paths(self):
        """Setup direktori untuk penyimpanan data"""
        base_path = Path(".")
        paths = self.config.get('paths', {})
        
        # Data directories
        self.raw_path = base_path / paths.get('data', {}).get('raw', 'data/raw')
        self.aggtrades_path = self.raw_path / "aggtrades"
        self.tmp_raw_path = self.raw_path / "tmp_raw" # Untuk ZIP mentah
        self.orderbook_path = base_path / paths.get('data', {}).get('orderbook', 'data/orderbook')
        self.unified_path = base_path / paths.get('data', {}).get('unified', 'data/unified')
        
        # Create directories
        for path in [self.raw_path, self.aggtrades_path, self.tmp_raw_path, self.orderbook_path, self.unified_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {path}")

    def _download_and_extract_vision(self, url: str, filename: str) -> Optional[pd.DataFrame]:
        """Helper untuk download, extract, dan parse file dari Binance Vision"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Binance vision zip biasanya berisi satu csv dengan nama yang sama
                    csv_filename = filename.replace(".zip", ".csv")
                    with z.open(csv_filename) as f:
                        # Klines: open_time, open, high, low, close, volume, close_time, quote_vol, count, taker_buy_base, taker_buy_quote, ignore
                        df = pd.read_csv(f, header=None, low_memory=False)
                        return df
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None

    def download_historical_klines(self, interval: str = "1m") -> pd.DataFrame:
        """
        Download data OHLCV historis dari Binance Data Vision (Asli, bukan dummy)
        """
        logger.info(f"Memulai download data historis {interval} dari Binance Vision")
        
        base_url = self.config.get('data_collection', {}).get('binance_data_vision_base',
                     "https://data.binance.vision/data/futures/um/daily")
        
        symbol = self.config['project']['symbol']
        all_dfs = []
        
        start_date = datetime.strptime(self.config['data_collection']['historical_start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(self.config['data_collection']['historical_end_date'], "%Y-%m-%d")
        
        total_days = (end_date - start_date).days + 1
        current_date = start_date
        pbar = tqdm(total=total_days, desc=f"Downloading {interval} klines")
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            year, month, day = current_date.strftime("%Y"), current_date.strftime("%m"), current_date.strftime("%d")
            
            filename = f"{symbol}-{interval}-{year}-{month}-{day}.zip"
            url = f"{base_url}/klines/{symbol}/{interval}/{filename}"
            
            df_day = self._download_and_extract_vision(url, filename)
            if df_day is not None:
                # Format: open_time, open, high, low, close, volume, ...
                df_day = df_day[[0, 1, 2, 3, 4, 5]]
                df_day.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                all_dfs.append(df_day)
            
            current_date += timedelta(days=1)
            pbar.update(1)
            time.sleep(0.1)
            
        pbar.close()
        
        if not all_dfs:
            logger.error(f"Tidak ada data klines {interval} yang berhasil diunduh")
            return pd.DataFrame()
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Bersihkan header jika ada (konversi ke numeric, yang gagal jadi NaN, lalu drop)
        for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            
        full_df = full_df.dropna(subset=['timestamp'])
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms')
        full_df.set_index('timestamp', inplace=True)
        
        # Simpan ke file
        output_file = self.raw_path / f"historical_{interval}_klines.parquet"
        full_df.to_parquet(output_file)
        logger.info(f"Data historis {interval} disimpan ke {output_file}")
        
        return full_df

    def _download_file_to_disk(self, url: str, target_path: Path) -> bool:
        """Helper untuk download file langsung ke disk"""
        if target_path.exists():
            return True
        try:
            response = self.session.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def download_historical_aggtrades(self, max_workers: int = 5) -> int:
        """
        Download aggTrades secara paralel (ke disk) lalu diproses sekuensial.
        """
        logger.info(f"Memulai download historical aggTrades (Parallel IO Mode, workers={max_workers})")
        
        base_url = self.config.get('data_collection', {}).get('binance_data_vision_base',
                     "https://data.binance.vision/data/futures/um/daily")
        symbol = self.config['project']['symbol']
        
        start_date = datetime.strptime(self.config['data_collection']['historical_start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(self.config['data_collection']['historical_end_date'], "%Y-%m-%d")
        
        # 1. Kumpulkan daftar tugas (dates yang belum ada parquet-nya)
        download_tasks = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            year, month, day = current_date.strftime("%Y"), current_date.strftime("%m"), current_date.strftime("%d")
            
            parquet_file = self.aggtrades_path / f"{symbol}-aggTrades-{date_str}.parquet"
            if not parquet_file.exists():
                zip_filename = f"{symbol}-aggTrades-{year}-{month}-{day}.zip"
                url = f"{base_url}/aggTrades/{symbol}/{zip_filename}"
                zip_path = self.tmp_raw_path / zip_filename
                download_tasks.append((url, zip_path, date_str, zip_filename))
            
            current_date += timedelta(days=1)
            
        if not download_tasks:
            logger.info("Semua data aggTrades sudah lengkap di disk.")
            return 0

        # 2. Parallel Download ZIP ke folder TMP
        logger.info(f"Mengunduh {len(download_tasks)} file ZIP secara paralel...")
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(self._download_file_to_disk, t[0], t[1]): t for t in download_tasks}
            pbar = tqdm(total=len(download_tasks), desc="Parallel Downloading")
            for future in concurrent.futures.as_completed(future_to_date):
                if future.result():
                    success_count += 1
                pbar.update(1)
            pbar.close()

        # 3. Sequential Processing (Hemat RAM)
        logger.info("Memulai pemrosesan ZIP ke Parquet (Sekuensial untuk hemat RAM)...")
        pbar = tqdm(total=len(download_tasks), desc="Processing ZIP to Parquet")
        for url, zip_path, date_str, zip_filename in download_tasks:
            if not zip_path.exists():
                pbar.update(1)
                continue
                
            parquet_file = self.aggtrades_path / f"{symbol}-aggTrades-{date_str}.parquet"
            try:
                with zipfile.ZipFile(zip_path) as z:
                    csv_filename = zip_filename.replace(".zip", ".csv")
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=None, low_memory=False)
                        df = df[[1, 2, 5, 6]]
                        df.columns = ['price', 'qty', 'timestamp', 'is_buyer_maker']
                        for col in ['price', 'qty', 'timestamp']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df.dropna(subset=['timestamp'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['price'] = df['price'].astype(float)
                        df['qty'] = df['qty'].astype(float)
                        df['is_buyer_maker'] = df['is_buyer_maker'].map({'true': True, 'false': False, '1': True, '0': False, True: True, False: False})
                        
                        df.to_parquet(parquet_file, compression='snappy', index=True)
                
                # Hapus ZIP setelah jadi Parquet untuk hemat disk
                zip_path.unlink()
                
                # Paksa GC
                del df
                gc.collect()
                
            except Exception as e:
                logger.error(f"Gagal memproses {zip_filename}: {e}")
                
            pbar.update(1)
        pbar.close()

        return success_count
    
    def download_funding_rate(self) -> pd.DataFrame:
        """Download data funding rate historis dengan iterasi periode"""
        logger.info("Memulai download funding rate historis (periode penuh)")
        
        base_url = self.config['data_collection']['binance_base_url']
        endpoint = self.config['data_collection']['endpoints']['funding_rate']
        symbol = self.config['project']['symbol']
        
        start_date = datetime.strptime(self.config['data_collection']['historical_start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(self.config['data_collection']['historical_end_date'], "%Y-%m-%d")
        
        current_start = int(start_date.timestamp() * 1000)
        final_end = int(end_date.timestamp() * 1000)
        
        all_dfs = []
        while current_start < final_end:
            url = f"{base_url}{endpoint}?symbol={symbol}&startTime={current_start}&limit=1000"
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if not data: break
                    df = pd.DataFrame(data)
                    all_dfs.append(df)
                    # Update start_time untuk batch berikutnya (tambah 1ms dari data terakhir)
                    current_start = int(data[-1]['fundingTime']) + 1
                    time.sleep(0.5)
                else:
                    break
            except Exception as e:
                logger.error(f"Error funding rate: {e}")
                break
                
        if not all_dfs: return pd.DataFrame()
        
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df['timestamp'] = pd.to_datetime(full_df['fundingTime'], unit='ms')
        full_df['funding_rate'] = pd.to_numeric(full_df['fundingRate'])
        full_df = full_df[['timestamp', 'funding_rate']].drop_duplicates()
        full_df.set_index('timestamp', inplace=True)
        
        # Potong data agar tidak melebihi end_date (23:59:59)
        cutoff_date = end_date.replace(hour=23, minute=59, second=59)
        full_df = full_df[full_df.index <= cutoff_date]
        
        output_file = self.raw_path / "historical_funding_rate.parquet"
        full_df.to_parquet(output_file)
        return full_df

    def download_open_interest(self) -> pd.DataFrame:
        """
        Download data Open Interest historis dari Binance Data Vision (Metrics)
        Catatan: Data ini diambil dari file harian metrics yang mengandung banyak kolom.
        """
        logger.info("Memulai download Open Interest historis dari Binance Vision (Metrics)")
        return self._download_metrics_from_vision("open_interest")

    def download_long_short_ratio(self) -> pd.DataFrame:
        """
        Download data Long/Short Ratio historis dari Binance Data Vision (Metrics)
        """
        logger.info("Memulai download Long/Short Ratio historis dari Binance Vision (Metrics)")
        return self._download_metrics_from_vision("long_short_ratio")

    def _download_metrics_from_vision(self, target_type: str) -> pd.DataFrame:
        """Helper untuk mengambil metrics harian (OI/LSR) dari Binance Vision"""
        output_file = self.raw_path / f"historical_{target_type}.parquet"
        
        # JANGAN download lagi jika sudah ada (sesuai permintaan user)
        if output_file.exists():
            logger.info(f"Memuat {target_type} dari disk (sudah ada).")
            return pd.read_parquet(output_file)

        base_url = "https://data.binance.vision/data/futures/um/daily/metrics"
        symbol = self.config['project']['symbol']
        
        start_date = datetime.strptime(self.config['data_collection']['historical_start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(self.config['data_collection']['historical_end_date'], "%Y-%m-%d")
        
        all_dfs = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        
        pbar = tqdm(total=total_days, desc=f"Downloading metrics for {target_type}")
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            year, month, day = current_date.strftime("%Y"), current_date.strftime("%m"), current_date.strftime("%d")
            
            filename = f"{symbol}-metrics-{year}-{month}-{day}.zip"
            url = f"{base_url}/{symbol}/{filename}"
            
            df_day = self._download_and_extract_vision(url, filename)
            if df_day is not None:
                try:
                    # Deteksi baris header
                    first_row = df_day.iloc[0].astype(str).tolist()
                    is_header = "create_time" in first_row or "symbol" in first_row
                    
                    if is_header:
                        df_day.columns = first_row
                        df_day = df_day.iloc[1:].reset_index(drop=True)
                        cols = df_day.columns.tolist()
                        
                        # create_time biasanya di kolom 0
                        time_col = cols[0]
                        
                        if target_type == "open_interest":
                            # Cari kolom 'sum_open_interest'
                            target_col = [c for c in cols if "sum_open_interest" in str(c).lower()][0]
                        else:
                            # Cari kolom 'count_long_short_ratio' atau 'global_long_short'
                            target_col = [c for c in cols if "long_short_ratio" in str(c).lower()][-1]
                        
                        temp_df = df_day[[time_col, target_col]].copy()
                        temp_df.columns = ['timestamp', target_type]
                    else:
                        # Fallback (indeks hasil observasi curl: 0=time, 2=OI, 6=LSR)
                        if target_type == "open_interest":
                            temp_df = df_day[[0, 2]].copy()
                        else:
                            temp_df = df_day[[0, 6]].copy()
                        
                        temp_df.columns = ['timestamp', target_type]
                    
                    all_dfs.append(temp_df)
                except Exception as e:
                    logger.warning(f"Gagal memproses metrics pada {date_str}: {e}")
            
            current_date += timedelta(days=1)
            pbar.update(1)
            time.sleep(0.05)
            
        pbar.close()
        
        if not all_dfs:
            logger.warning(f"Tidak ada data terkumpul untuk {target_type}")
            return pd.DataFrame()
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Format Timestamp (Bisa berupa String "2025-04-29 00:00:00" atau Numeric ms)
        try:
            # Coba konversi ke numeric dulu (jika MS)
            temp_ts = pd.to_numeric(full_df['timestamp'], errors='coerce')
            if temp_ts.notna().sum() > (len(full_df) * 0.5):
                full_df['timestamp'] = pd.to_datetime(temp_ts, unit='ms')
            else:
                # Jika gagal, coba parsing sebagai string datetime (Binance Vision format)
                full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], errors='coerce')
        except:
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], errors='coerce')
        
        # Clean data kolom target
        col_to_check = target_type
        full_df[col_to_check] = pd.to_numeric(full_df[col_to_check], errors='coerce')
        
        full_df = full_df.dropna(subset=['timestamp', col_to_check])
        full_df.set_index('timestamp', inplace=True)
        
        # Potong data agar tidak melebihi end_date (23:59:59)
        cutoff_date = end_date.replace(hour=23, minute=59, second=59)
        full_df = full_df[full_df.index <= cutoff_date]
        
        # Simpan
        full_df.to_parquet(output_file)
        logger.info(f"Data {target_type} BERHASIL disimpan ke {output_file} dengan {len(full_df)} baris")
        
        return full_df
    
    def validate_data(self, df: pd.DataFrame, data_type: str) -> bool:
        """
        Validasi data sebelum disimpan
        
        Args:
            df: DataFrame untuk divalidasi
            data_type: Tipe data (klines, funding, oi, lsr)
            
        Returns:
            True jika valid, False jika tidak
        """
        if df.empty:
            logger.warning(f"Data {data_type} kosong")
            return False
        
        # Check for required columns based on data type
        if data_type == "klines":
            required_cols = ['open', 'high', 'low', 'close', 'volume']
        elif data_type == "funding":
            required_cols = ['funding_rate']
        elif data_type == "oi":
            required_cols = ['open_interest']
        elif data_type == "lsr":
            required_cols = ['long_short_ratio']
        else:
            logger.error(f"Tipe data tidak dikenal: {data_type}")
            return False
        
        # Check columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Kolom hilang untuk {data_type}: {missing_cols}")
            return False
        
        # Check for NaN values
        nan_count = df[required_cols].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data {data_type} memiliki {nan_count} nilai NaN")
        
        # Check for negative values where not allowed
        if data_type == "klines":
            negative_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in negative_cols:
                if (df[col] < 0).any():
                    logger.warning(f"Data {data_type} memiliki nilai negatif di kolom {col}")
        
        logger.info(f"Data {data_type} valid dengan {len(df)} baris")
        return True
    
    def get_realtime_orderbook(self, limit: int = 20) -> dict:
        """
        Ambil snapshot orderbook real-time dari Binance
        
        Args:
            limit: Jumlah level bid/ask (default: 20 sesuai config)
            
        Returns:
            Dictionary dengan orderbook data
        """
        base_url = self.config.get('data_collection', {}).get('binance_base_url',
                     "https://fapi.binance.com")
        symbol = self.config['project']['symbol']
        
        url = f"{base_url}/fapi/v1/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Tambahkan timestamp
            data['timestamp'] = int(time.time() * 1000)
            data['local_timestamp'] = datetime.now().isoformat()
            
            logger.debug(f"Orderbook snapshot diambil: {data['timestamp']}")
            return data
            
        except Exception as e:
            logger.error(f"Error mengambil orderbook: {e}")
            return {}
    
    def get_realtime_klines(self, interval: str = "1m", limit: int = 1) -> list:
        """
        Ambil data klines real-time terbaru
        
        Args:
            interval: Interval candle (1m, 15m)
            limit: Jumlah candle yang diambil
            
        Returns:
            List of klines data
        """
        base_url = self.config.get('data_collection', {}).get('binance_base_url',
                     "https://fapi.binance.com")
        symbol = self.config['project']['symbol']
        
        url = f"{base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            logger.debug(f"Klines {interval} real-time diambil: {len(data)} candle")
            return data
            
        except Exception as e:
            logger.error(f"Error mengambil klines real-time: {e}")
            return []
    
    def get_realtime_ticker(self) -> dict:
        """
        Ambil ticker price 24h real-time
        
        Returns:
            Dictionary dengan ticker data
        """
        base_url = self.config.get('data_collection', {}).get('binance_base_url',
                     "https://fapi.binance.com")
        symbol = self.config['project']['symbol']
        
        url = f"{base_url}/fapi/v1/ticker/24hr"
        params = {'symbol': symbol}
        
        try:
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            logger.debug(f"Ticker real-time diambil: price={data.get('lastPrice')}")
            return data
            
        except Exception as e:
            logger.error(f"Error mengambil ticker real-time: {e}")
            return {}
    
    def collect_realtime_snapshot(self) -> dict:
        """
        Kumpulkan semua data real-time dalam satu snapshot
        Mengikuti rate limit: 1 request per detik
        
        Returns:
            Dictionary dengan semua data real-time
        """
        snapshot = {
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.now().isoformat()
        }
        
        # Ambil orderbook
        orderbook = self.get_realtime_orderbook()
        if orderbook:
            snapshot['orderbook'] = orderbook
        
        # Tunggu untuk rate limit
        time.sleep(1.0 / self.config.get('data_collection', {}).get('requests_per_second', 1.0))
        
        # Ambil klines 1m
        klines_1m = self.get_realtime_klines(interval="1m", limit=5)
        if klines_1m:
            snapshot['klines_1m'] = klines_1m
        
        # Tunggu untuk rate limit
        time.sleep(1.0 / self.config.get('data_collection', {}).get('requests_per_second', 1.0))
        
        # Ambil klines 15m
        klines_15m = self.get_realtime_klines(interval="15m", limit=5)
        if klines_15m:
            snapshot['klines_15m'] = klines_15m
        
        # Ambil ticker
        ticker = self.get_realtime_ticker()
        if ticker:
            snapshot['ticker'] = ticker
        
        logger.info(f"Snapshot real-time dikumpulkan: {snapshot['timestamp']}")
        return snapshot
    
    def save_realtime_snapshot(self, snapshot: dict):
        """
        Simpan snapshot real-time ke file parquet
        
        Args:
            snapshot: Data snapshot dari collect_realtime_snapshot()
        """
        if not snapshot:
            logger.warning("Snapshot kosong, tidak disimpan")
            return
        
        # Buat DataFrame dari snapshot
        timestamp = snapshot['timestamp']
        date_str = datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d")
        
        # Simpan ke file parquet
        filename = f"realtime_{timestamp}.parquet"
        filepath = self.orderbook_path / date_str / filename
        
        # Buat direktori jika belum ada
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert snapshot to DataFrame-friendly format
        import json
        snapshot_str = json.dumps(snapshot)
        
        df = pd.DataFrame([{
            'timestamp': timestamp,
            'snapshot': snapshot_str,
            'date': date_str
        }])
        
        df.to_parquet(filepath, compression='snappy')
        logger.debug(f"Snapshot disimpan: {filepath}")
    
    def run_realtime_collection(self, duration_minutes: int = 60):
        """
        Jalankan koleksi data real-time untuk durasi tertentu
        
        Args:
            duration_minutes: Durasi koleksi dalam menit
        """
        logger.info(f"=== MEMULAI KOLEKSI DATA REAL-TIME UNTUK {duration_minutes} MENIT ===")
        
        end_time = time.time() + (duration_minutes * 60)
        snapshot_count = 0
        
        try:
            while time.time() < end_time:
                # Kumpulkan snapshot
                snapshot = self.collect_realtime_snapshot()
                
                # Simpan snapshot
                self.save_realtime_snapshot(snapshot)
                snapshot_count += 1
                
                # Tunggu sampai interval berikutnya (default: 60 detik)
                save_interval = self.config.get('data_collection', {}).get('save_interval_seconds', 60)
                time.sleep(save_interval)
                
                # Log progress setiap 5 snapshot
                if snapshot_count % 5 == 0:
                    remaining = int((end_time - time.time()) / 60)
                    logger.info(f"Progress: {snapshot_count} snapshot, {remaining} menit tersisa")
        
        except KeyboardInterrupt:
            logger.info("Koleksi data real-time dihentikan oleh user")
        except Exception as e:
            logger.error(f"Error dalam koleksi real-time: {e}")
        finally:
            logger.info(f"=== KOLEKSI DATA REAL-TIME SELESAI ===")
            logger.info(f"Total snapshot dikumpulkan: {snapshot_count}")
    
    def run_historical_download(self):
        """Jalankan semua download data historis"""
        logger.info("=== MEMULAI DOWNLOAD DATA HISTORIS ===")
        
        # Download OHLCV data
        klines_1m = self.download_historical_klines(interval="1m")
        # klines_15m = self.download_historical_klines(interval="15m")
        
        # Load existing klines instead of downloading
        # try:
        #     klines_1m = pd.read_parquet(self.raw_path / "historical_1m_klines.parquet")
        #     klines_15m = pd.read_parquet(self.raw_path / "historical_15m_klines.parquet")
        #     logger.info("Berhasil memuat klines 1m dan 15m dari disk.")
        # except Exception as e:
        #     logger.warning(f"Gagal memuat klines dari disk, menggunakan DataFrame kosong: {e}")
        #     klines_1m = pd.DataFrame()
        #     klines_15m = pd.DataFrame()
        
        # Download AggTrades (Proxy OB) - Sekarang disimpan per hari
        # aggtrades_count = self.download_historical_aggtrades()
        
        # Download supplementary data
        # funding_rate = self.download_funding_rate()
        # open_interest = self.download_open_interest()
        # long_short_ratio = self.download_long_short_ratio()
        
        # Validasi data
        self.validate_data(klines_1m, "klines")
        # self.validate_data(klines_15m, "klines")
        # # Validasi untuk aggtrades di-skip di sini karena sudah divalidasi per hari
        # self.validate_data(funding_rate, "funding")
        # self.validate_data(open_interest, "oi")
        # self.validate_data(long_short_ratio, "lsr")
        
        logger.info("=== DOWNLOAD DATA HISTORIS SELESAI ===")
        
        # Return summary
        return {
            'klines_1m': len(klines_1m),
            # 'klines_15m': len(klines_15m),
            # 'aggtrades_files': aggtrades_count,
            # 'funding_rate': len(funding_rate),
            # 'open_interest': len(open_interest),
            # 'long_short_ratio': len(long_short_ratio)
        }


def main():
    """Fungsi utama"""
    parser = argparse.ArgumentParser(description='Data Collection untuk Polymarket BTC 15-Minute Prediction')
    parser.add_argument('--config', type=str, default='/run/media/rotan/New Volume/gemini3/orderbook-kline-aggtrade/config.yaml', help='Path ke file konfigurasi')
    parser.add_argument('--mode', type=str, default='historical',
                       choices=['historical', 'realtime', 'all'], help='Mode operasi')
    parser.add_argument('--duration', type=int, default=60,
                       help='Durasi koleksi real-time dalam menit (default: 60)')
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Initialize collector
    collector = BinanceDataCollector(args.config)
    
    if args.mode in ['historical', 'all']:
        # Download historical data
        summary = collector.run_historical_download()
        logger.info(f"Summary: {summary}")
    
    if args.mode in ['realtime', 'all']:
        # Jalankan koleksi real-time
        logger.info(f"Memulai koleksi data real-time untuk {args.duration} menit")
        collector.run_realtime_collection(duration_minutes=args.duration)
    
    logger.info("Data collection selesai")


if __name__ == "__main__":
    main()