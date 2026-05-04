import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Local imports
from inference_v1 import PolymarketPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically find project root (assuming script is in src/inference/backtest_us_session.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_us_session_backtest(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.yaml")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Paths & Predictor
    # Resolve artifacts path
    artifacts_rel = config.get('paths', {}).get('models', {}).get('artifacts', "models/artifacts")
    artifacts_path = str(PROJECT_ROOT / artifacts_rel)
    predictor = PolymarketPredictor(artifacts_path, config_path)
    
    # 1. Load Data
    features_rel = Path(config['paths']['data']['features_v1'])
    features_path = PROJECT_ROOT / features_rel / "final_features.parquet"
    df = pd.read_parquet(features_path).sort_values('round_start')
    
    # Use Test set
    n = len(df)
    val_end = int(n * (config['training']['train_ratio'] + config['training']['val_ratio']))
    test_df = df.iloc[val_end:].copy()
    
    if 'round_start' not in test_df.columns:
        test_df = test_df.reset_index()
    
    # 2. FILTER SESI US & OVERLAP (13:00 - 21:00 UTC)
    test_df['round_start'] = pd.to_datetime(test_df['round_start'])
    hours = test_df['round_start'].dt.hour
    
    # Definisi sesi: 13:00 s/d 20:45 (masuk ke jam 21)
    mask_us = (hours >= 13) & (hours < 21)
    test_df = test_df[mask_us].copy()
    
    logger.info(f"Filtered for US/Overlap Session. Rounds: {len(test_df)}")
    if len(test_df) == 0:
        logger.error("No data found for the specified session hours.")
        return

    # 3. Generate Probabilities
    test_df['p_model'] = predictor.predict_probability(test_df)
    
    # 4. Simulate Market Price (More Conservative noise)
    np.random.seed(42)
    test_df['p_market'] = 0.5 + np.random.normal(0, 0.015, len(test_df))
    test_df['p_market'] = test_df['p_market'].clip(0.40, 0.60)

    # 5. Trading Simulation
    initial_bankroll = config['trade_decision']['initial_bankroll']
    bankroll = initial_bankroll
    equity_curve = [initial_bankroll]
    trades = []
    
    for idx, row in test_df.iterrows():
        # Simulate Spread (1.5% - 3.5%)
        spread = np.random.uniform(0.015, 0.035)
        p_market_mid = row['p_market']
        best_bid = p_market_mid - spread/2
        best_ask = p_market_mid + spread/2

        decision = predictor.get_trade_decision(row['p_model'], best_bid, best_ask)

        if decision['decision'] != 'SKIP':
            bet_amount = bankroll * decision['bet_size']

            is_win = (decision['decision'] == 'BET_YES' and row['target'] == 1) or \
                     (decision['decision'] == 'BET_NO' and row['target'] == 0)

            exec_p = decision['execution_price']
            if is_win:
                # Menghitung profit berdasarkan harga eksekusi real
                shares = bet_amount / exec_p
                net_profit = (shares * 1.0 - bet_amount) * (1 - config['trade_decision']['polymarket_fee'])
                bankroll += net_profit
            else:
                net_profit = -bet_amount
                bankroll += net_profit

            trades.append({
                'round_start': row['round_start'],
                'decision': decision['decision'],
                'p_model': row['p_model'],
                'p_market': p_market_mid,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'execution_price': exec_p,
                'is_win': is_win,
                'net_profit': net_profit,
                'bankroll': bankroll
            })
            
        equity_curve.append(bankroll)

    # 6. Reporting
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        logger.warning("No trades executed. Try lowering min_ev_to_trade in config.yaml.")
        return

    win_rate = trades_df['is_win'].mean()
    total_profit_pct = (bankroll - initial_bankroll) / initial_bankroll
    
    # Sharpe
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = np.sqrt(365 * (len(test_df)/len(df.iloc[val_end:])) * 96) * returns.mean() / (returns.std() + 1e-9)

    logger.info(f"\n--- US SESSION BACKTEST RESULTS ---")
    logger.info(f"Total Session Rounds : {len(test_df)}")
    logger.info(f"Trades Executed      : {len(trades_df)} (Cov: {len(trades_df)/len(test_df):.1%})")
    logger.info(f"Win Rate             : {win_rate:.2%}")
    logger.info(f"Final Bankroll       : ${bankroll:.2f}")
    logger.info(f"Profit %             : {total_profit_pct:.1%}")
    logger.info(f"Sharpe Ratio         : {sharpe:.2f}")

    # Plotting
    output_rel = Path(config['paths']['reports'])
    output_dir = PROJECT_ROOT / output_rel
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"US Session Equity Curve - {config['project']['name']}")
    plt.xlabel("Trades")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.savefig(output_dir / "equity_curve_us_session.png")
    logger.info(f"Visual report saved to: {output_dir / 'equity_curve_us_session.png'}")

if __name__ == "__main__":
    run_us_session_backtest()
