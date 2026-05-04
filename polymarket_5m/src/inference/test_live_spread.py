import sys
from pathlib import Path
import logging

# Set path agar bisa import dari src/inference
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src/inference"))

from find_markets import PolymarketMarketFinder

# Setup logging sederhana
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_live_spread():
    print("\n" + "="*50)
    print("POLYMARKET LIVE SPREAD TESTER")
    print("="*50)
    
    market = PolymarketMarketFinder.find_bitcoin_price_market()
    
    if not market:
        print("\n[!] GAGAL: Tidak ada pasar 15m BTC yang aktif atau likuid saat ini.")
        return

    bid = market['best_bid']
    ask = market['best_ask']
    spread = ask - bid
    mid = (bid + ask) / 2
    
    print(f"\nMarket Found: {market['question']}")
    print(f"Condition ID: {market['id']}")
    print(f"Ends At     : {market['end_date']}")
    
    print("\n" + "-"*30)
    print(f"Best BID (NO) : {bid:.4f}")
    print(f"Best ASK (YES): {ask:.4f}")
    print(f"Mid-Price     : {mid:.4f}")
    print("-"*30)
    
    print(f"REAL SPREAD   : {spread:.4f} ({spread*100:.2f}%)")
    
    # Meniru logika safeguard di shadow_trader.py
    print("\n[SHADOW TRADER SAFEGUARD CHECK]")
    if bid <= 0.0 or ask >= 1.0 or ask <= bid:
        print("Status: REJECTED (Invalid market data)")
    elif spread > 0.05:
        print(f"Status: REJECTED (Spread {spread:.4f} > 0.05)")
    else:
        print("Status: ACCEPTED (Market is liquid enough)")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_live_spread()
