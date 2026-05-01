import pandas as pd
import numpy as np
import joblib
import yaml
import logging
import sys
from pathlib import Path
import catboost as cb

# Menambahkan folder models ke path agar kelas RegimeDetector bisa ditemukan saat joblib.load
sys.path.append("/run/media/rotan/New Volume/gemini3/polymarket_5m/src/models")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolymarketPredictor:
    def __init__(self, artifacts_path: str, config_path: str = "/run/media/rotan/New Volume/gemini3/polymarket_5m/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.artifacts_path = Path(artifacts_path)
        
        # Load artifacts
        logger.info("Loading models and artifacts...")
        self.cb_model = cb.CatBoostClassifier()
        self.cb_model.load_model(str(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/catboost_final_v1.cbm"))
        
        self.lgb_model = joblib.load(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/lgbm_final_v1.pkl")
        self.calibrator = joblib.load(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/calibrator_v1.pkl")
        self.ensemble_weights = joblib.load(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/ensemble_weights_v1.pkl")
        self.best_t = joblib.load(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/optimal_threshold_v1.pkl")
        self.selected_features = joblib.load(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/selected_features_v1.pkl")
        
        try:
            # Perbaikan import regime_detector (asumsi ada di PYTHONPATH atau folder yang sama)
            from regime_detector import RegimeDetector
            self.regime_detector = RegimeDetector.load(str(self.artifacts_path / "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts/regime_detector_v1.pkl"))
        except:
            logger.warning("Regime detector not found or failed to load.")
            self.regime_detector = None

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate calibrated ensemble probability
        """
        # Ensure only selected features are used
        X_sub = X[self.selected_features].astype(np.float32)
        
        p_cb = self.cb_model.predict_proba(X_sub)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X_sub)[:, 1]
        
        w_cb, w_lgb = self.ensemble_weights
        p_ens = p_cb * w_cb + p_lgb * w_lgb
        
        return self.calibrator.predict(p_ens)

    def get_trade_decision(self, p_model: float, p_market: float) -> dict:
        """
        Formula EV: p_model - p_market - fee - margin
        """
        fee    = self.config['trade_decision']['polymarket_fee']
        margin = self.config['trade_decision']['model_risk_margin']
        min_ev = self.config['trade_decision']['min_ev_to_trade']

        # Edge untuk YES
        ev_yes = p_model - p_market - fee - margin

        # Edge untuk NO
        p_model_no  = 1 - p_model
        p_market_no = 1 - p_market
        ev_no       = p_model_no - p_market_no - fee - margin

        if ev_yes > min_ev and ev_yes >= ev_no:
            decision = 'BET_YES'
            ev = ev_yes
            confidence = p_model
        elif ev_no > min_ev and ev_no > ev_yes:
            decision = 'BET_NO'
            ev = ev_no
            confidence = p_model_no
        else:
            decision = 'SKIP'
            ev = max(ev_yes, ev_no)
            confidence = abs(p_model - 0.5)

        # Bet sizing (Kelly)
        bet_size = 0
        if decision != 'SKIP':
            b = (1 / p_market if decision == 'BET_YES' else 1 / p_market_no) - 1
            p_win = p_model if decision == 'BET_YES' else p_model_no
            kelly_full = (p_win * b - (1 - p_win)) / (b + 1e-9)
            kelly_frac = kelly_full * self.config['trade_decision']['kelly_fraction']
            bet_size = max(0, min(kelly_frac, self.config['trade_decision']['max_bet_fraction']))

        return {
            'decision': decision,
            'ev': ev,
            'bet_size': bet_size,
            'confidence': confidence,
            'p_model': p_model,
            'p_market': p_market
        }

if __name__ == "__main__":
    # Example usage
    artifacts = "/run/media/rotan/New Volume/gemini3/polymarket_5m/models/artifacts"
    predictor = PolymarketPredictor(artifacts)
    logger.info("Predictor initialized and ready.")
