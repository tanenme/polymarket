import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src/inference"))
sys.path.append(str(PROJECT_ROOT / "src/models"))

from shadow_trader import ShadowTrader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_imaginary_simulation():
    # 1. Inisialisasi ShadowTrader (untuk akses pipeline fitur)
    trader = ShadowTrader()
    
    print("\n" + "="*60)
    print("SIMULASI TRADE IMAJINER (REAL-TIME FEATURES + LIVE ORDERBOOK)")
    print("="*60)
    
    # 2. Sync data Bybit terbaru untuk generate features
    print("[1/4] Mengambil data Bybit (BTCUSDT) untuk input model...")
    trader.sync_data() # Mengambil klines & aggTrades terbaru
    
    # 3. Ambil data Order Book live
    print("[2/4] Mengambil data Order Book live dari Polymarket...")
    best_bid, best_ask, p_market_mid, desc = trader.get_polymarket_data()
    
    if best_bid == 0:
        print("[!] GAGAL: Tidak bisa mengambil data pasar.")
        return
        
    print(f"      Best Bid: {best_bid:.4f} | Best Ask: {best_ask:.4f} | Mid: {p_market_mid:.4f}")
    print(f"      Spread  : {best_ask - best_bid:.4f}")
    
    # 4. Jalankan Pipeline Prediksi
    print("[3/4] Menghitung probabilitas menggunakan Ensemble Model...")
    
    # Logic copied from shadow_trader.run_prediction
    obi = (best_bid - (1 - best_ask))
    trader.buffer_1m.iloc[-1, trader.buffer_1m.columns.get_loc('obi_mean')] = obi
    
    df_1m = trader.add_technical_features_v1(trader.buffer_1m)
    df_1m = trader.add_time_features_v1(df_1m)
    df_aggr = trader.aggregate_window_features_v1(df_1m.tail(15))
    df_final = trader.add_inter_round_features_v1(df_aggr)
    
    X = df_final[trader.predictor.selected_features]
    p_model = trader.predictor.predict_probability(X)[0]
    
    # 5. Ambil Keputusan Trade
    print("[4/4] Menghitung Expected Value (EV) & Kelly Sizing...")
    decision = trader.predictor.get_trade_decision(p_model, best_bid, best_ask)
    
    print("\n" + "-"*40)
    print(f"HASIL PREDIKSI MODEL")
    print(f"Probabilitas Model (UP): {p_model:.4f}")
    print(f"Probabilitas Pasar (Mid): {p_market_mid:.4f}")
    print(f"Edge/EV Real          : {decision['ev']:.4f}")
    print("-"*40)
    
    print(f"\nKEPUTUSAN: {decision['decision']}")
    if decision['decision'] != 'SKIP':
        print(f"Eksekusi di Harga : {decision['execution_price']:.2f}")
        print(f"Bet Size (Kelly)  : {decision['bet_size']:.2%}")
        amount = trader.bankroll * decision['bet_size']
        print(f"Estimasi Taruhan  : ${amount:.2f} dari Bankroll ${trader.bankroll:.2f}")
    else:
        print("Alasan SKIP: EV tidak memenuhi target minimum atau margin risiko.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run_imaginary_simulation()
