"""
Polymarket 15M Bitcoin Market Finder
Uses Tag-based discovery for high precision.
"""

from __future__ import annotations
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolymarketMarketFinder:
    """Finder specifically for the 15-minute BTC Up/Down recurring markets."""
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    TAG_15M = "102467" # Tag ID for 15M markets

    @classmethod
    def find_bitcoin_price_market(cls) -> dict | None:
        """
        Discovery logic:
        1. Fetch all active markets with the 15M tag.
        2. Filter for those with 'Bitcoin Up or Down' in the question.
        3. Pick the one that is currently active and has liquidity.
        """
        try:
            # Step 1: Get active markets by Tag
            url = f"{cls.GAMMA_URL}/markets?active=true&tag_id={cls.TAG_15M}&limit=100"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code != 200:
                logger.error(f"Gamma API error: {resp.status_code}")
                return None
                
            markets = resp.json()
            
            for m in markets:
                question = m.get('question', '')
                
                # Check if it's the correct Bitcoin Up/Down market
                if "bitcoin up or down" in question.lower():
                    tokens = m.get('clobTokenIds')
                    if isinstance(tokens, str):
                        import json
                        tokens = json.loads(tokens)
                    
                    if not tokens or len(tokens) < 2:
                        continue
                    
                    yes_token = tokens[0]
                    
                    # Step 2: Check live Order Book for liquidity
                    book_url = f"{cls.CLOB_URL}/book?token_id={yes_token}"
                    book_resp = requests.get(book_url, timeout=5)
                    
                    if book_resp.status_code == 200:
                        book = book_resp.json()
                        bids = book.get('bids', [])
                        asks = book.get('asks', [])
                        
                        # Market is valid if it has at least one bid and one ask
                        if bids and asks:
                            # CLOB API often returns sorted lists, but to be 100% safe, 
                            # we find the max bid (best price for seller) and min ask (best price for buyer)
                            bid_prices = [float(b['price']) for b in bids]
                            ask_prices = [float(ask['price']) for ask in asks]
                            
                            best_bid = max(bid_prices)
                            best_ask = min(ask_prices)
                            
                            logger.info(f"Successfully discovered active market: {question}")
                            return {
                                'id': m.get('conditionId'),
                                'question': question,
                                'yes_token': yes_token,
                                'no_token': tokens[1],
                                'best_bid': best_bid,
                                'best_ask': best_ask,
                                'end_date': m.get('endDate')
                            }
            
            logger.warning("No active 15m Bitcoin Up/Down markets with liquidity found.")
            return None
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return None

# Module-level variable
ACTIVE_MARKET = PolymarketMarketFinder.find_bitcoin_price_market()

if __name__ == "__main__":
    market = PolymarketMarketFinder.find_bitcoin_price_market()
    if market:
        print("\n--- FOUND ACTIVE 15M BTC MARKET ---")
        print(f"Question: {market['question']}")
        print(f"YES Token: {market['yes_token']}")
        print(f"Ends At: {market['end_date']}")
        print(f"Market Price (Mid): {(market['best_bid'] + market['best_ask'])/2:.3f}")
    else:
        print("Failure: Could not find an active 15m Bitcoin market.")
