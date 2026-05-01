import pandas as pd
import numpy as np
import joblib
import yaml
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Local imports - Fixed import to use the new module name
from inference_v1 import PolymarketPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest_v1(config_path: str = "/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Use hardcoded path as requested for artifacts
    artifacts_path = "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts"
    predictor = PolymarketPredictor(artifacts_path, config_path)
    
    # 1. Load Test Data
    features_path = Path(config['paths']['data']['features_v1']) / "final_features.parquet"
    df = pd.read_parquet(features_path).sort_values('round_start')
    
    # Use Test set only
    n = len(df)
    val_end = int(n * (config['training']['train_ratio'] + config['training']['val_ratio']))
    test_df = df.iloc[val_end:].copy()
    
    # Ensure round_start is a column for iteration
    if 'round_start' not in test_df.columns:
        test_df = test_df.reset_index()
    
    logger.info(f"Running backtest on {len(test_df)} rounds...")
    
    # 2. Generate Probabilities
    test_df['p_model'] = predictor.predict_probability(test_df)
    
    # 3. Simulate Market Price (Approximation)
    np.random.seed(42)
    test_df['p_market'] = 0.5 + np.random.normal(0, 0.02, len(test_df))
    test_df['p_market'] = test_df['p_market'].clip(0.1, 0.9)

    # 4. Trading Simulation
    initial_bankroll = 70.0
    bankroll = initial_bankroll
    equity_curve = [initial_bankroll]
    trades = []
    
    for idx, row in test_df.iterrows():
        decision = predictor.get_trade_decision(row['p_model'], row['p_market'])
        
        if decision['decision'] != 'SKIP':
            bet_amount = bankroll * decision['bet_size']
            
            # Outcome
            is_win = (decision['decision'] == 'BET_YES' and row['target'] == 1) or \
                     (decision['decision'] == 'BET_NO' and row['target'] == 0)
            
            if is_win:
                p_m = row['p_market'] if decision['decision'] == 'BET_YES' else (1 - row['p_market'])
                gross_profit = bet_amount * (1/p_m - 1)
                net_profit = gross_profit * (1 - config['trade_decision']['polymarket_fee'])
                bankroll += net_profit
            else:
                net_profit = -bet_amount
                bankroll += net_profit
                
            trades.append({
                'round_start': row['round_start'],
                'decision': decision['decision'],
                'p_model': row['p_model'],
                'p_market': row['p_market'],
                'bet_size_pct': decision['bet_size'],
                'bet_amount': bet_amount,
                'net_profit': net_profit,
                'bankroll': bankroll,
                'is_win': is_win
            })
            
        equity_curve.append(bankroll)

    # 5. Performance Metrics
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        logger.warning("No trades executed during backtest. Check threshold/EV logic.")
        return

    total_profit = bankroll - initial_bankroll
    win_rate = trades_df['is_win'].mean()
    n_trades = len(trades_df)
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = np.sqrt(365 * 96) * returns.mean() / (returns.std() + 1e-9)
    
    equity_ser = pd.Series(equity_curve)
    drawdown = (equity_ser.cummax() - equity_ser) / equity_ser.cummax()
    max_dd = drawdown.max()

    logger.info(f"\n--- BACKTEST RESULTS ---")
    logger.info(f"Total Rounds   : {len(test_df)}")
    logger.info(f"Total Trades   : {n_trades} (Cov: {n_trades/len(test_df):.1%})")
    logger.info(f"Win Rate       : {win_rate:.2%}")
    logger.info(f"Final Bankroll : ${bankroll:.2f}")
    logger.info(f"Total Profit   : ${total_profit:.2f} ({total_profit/initial_bankroll:.1%})")
    logger.info(f"Max Drawdown   : {max_dd:.2%}")
    logger.info(f"Sharpe Ratio   : {sharpe:.2f}")

    # 6. Save and Plot
    # Pastikan jalur absolut untuk root proyek
    project_root = Path("/run/media/rotan/New Volume/gemini3/polymarket_5m")
    output_dir = project_root / config['paths']['reports']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Equity Curve - {config['project']['name']}")
    plt.xlabel("Rounds")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.savefig(output_dir / "equity_curve.png")
    logger.info(f"Report and Equity Curve saved to {output_dir}")

if __name__ == "__main__":
    run_backtest_v1()
