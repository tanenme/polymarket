import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import catboost as cb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recalibrate_v1(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    artifacts_path = Path(config['paths']['models']['artifacts'])
    
    # 1. Load data
    features_path = Path(config['paths']['data']['features_v1']) / "final_features.parquet"
    df = pd.read_parquet(features_path).sort_values('round_start')
    
    # Use Validation set for calibration
    n = len(df)
    train_end = int(n * config['training']['train_ratio'])
    val_end = train_end + int(n * config['training']['val_ratio'])
    val_df = df.iloc[train_end:val_end]
    
    # 2. Load models
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(artifacts_path / "catboost_final_v1.cbm"))
    lgb_model = joblib.load(artifacts_path / "lgbm_final_v1.pkl")
    ensemble_weights = joblib.load(artifacts_path / "ensemble_weights_v1.pkl")
    selected_features = joblib.load(artifacts_path / "selected_features_v1.pkl")
    
    X_val = val_df[selected_features].astype(np.float32)
    y_val = val_df['target']
    
    # 3. Generate probabilities
    p_cb = cb_model.predict_proba(X_val)[:, 1]
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    
    w_cb, w_lgb = ensemble_weights
    p_ens = p_cb * w_cb + p_lgb * w_lgb
    
    # 4. Check calibration before
    fraction_pos, mean_pred = calibration_curve(y_val, p_ens, n_bins=10)
    ece_before = np.mean(np.abs(fraction_pos - mean_pred))
    brier_before = brier_score_loss(y_val, p_ens)
    logger.info(f"Before Recalibration: ECE={ece_before:.4f}, Brier={brier_before:.4f}")
    
    # 5. Fit new calibrator
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_ens, y_val)
    
    # 6. Check calibration after
    p_cal = calibrator.predict(p_ens)
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_val, p_cal, n_bins=10)
    ece_after = np.mean(np.abs(fraction_pos_cal - mean_pred_cal))
    brier_after = brier_score_loss(y_val, p_cal)
    logger.info(f"After Recalibration: ECE={ece_after:.4f}, Brier={brier_after:.4f}")
    
    # Save new calibrator
    joblib.dump(calibrator, artifacts_path / "calibrator_v1.pkl")
    logger.info(f"New calibrator saved to {artifacts_path / 'calibrator_v1.pkl'}")

if __name__ == "__main__":
    recalibrate_v1()
