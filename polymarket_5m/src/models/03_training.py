import pandas as pd
import numpy as np
import logging
import yaml
import joblib
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import catboost as cb
import optuna
from datetime import datetime

# Local imports
from regime_detector import RegimeDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically find project root (assuming script is in src/models/03_training.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_and_validate_data_v1(features_path: str, config: dict) -> pd.DataFrame:
    logger.info(f"Loading data from {features_path}...")
    df = pd.read_parquet(features_path)

    if 'round_start' not in df.columns:
        df = df.reset_index()

    df = df.sort_values('round_start').reset_index(drop=True)

    # STEP 1: Hitung gap antar round (dalam detik)
    df['gap_to_prev'] = pd.to_datetime(df['round_start']).diff().dt.total_seconds()

    # STEP 2: Flag round yang round sebelumnya ada gap besar (> 30 menit)
    df['has_valid_context'] = (df['gap_to_prev'] <= 1800) | df['gap_to_prev'].isna()

    # STEP 3: Invalidasi inter-round features untuk round tanpa konteks valid
    lag_cols = [c for c in df.columns if c.startswith('past_')]
    n_invalidated = (~df['has_valid_context']).sum()
    df.loc[~df['has_valid_context'], lag_cols] = np.nan
    logger.info(f"Invalidated {n_invalidated} rounds' inter-round features (gap > 30min)")

    # STEP 4: Buang round tanpa target
    df = df.dropna(subset=['target', 'future_return'])

    # STEP 5: Sanity checks
    n = len(df)
    up_rate = df['target'].mean()
    
    # Calculate coverage
    total_seconds = (pd.to_datetime(df['round_start'].max()) - pd.to_datetime(df['round_start'].min())).total_seconds()
    expected_rounds = int(total_seconds / 900) + 1
    coverage_overall = n / max(1, expected_rounds)

    logger.info(f"Loaded: {n} rounds | UP rate: {up_rate:.2%} | Coverage: {coverage_overall:.1%}")

    if up_rate < 0.38 or up_rate > 0.62:
        logger.warning(f"UP rate {up_rate:.2%} is outside normal range [38%, 62%].")

    return df

def is_valid_feature(col_name: str, config: dict) -> bool:
    exclude = set(config['features']['exclude_cols'])
    price_abs = set(config['features']['price_absolute_cols'])
    
    if col_name in exclude: return False
    if col_name in price_abs: return False
    if col_name.startswith('snap_') and any(p in col_name for p in price_abs): return False
    
    return True

def compute_sample_weights(train_df: pd.DataFrame, config: dict) -> np.ndarray:
    w_high = config['training']['sample_weight_high']
    w_low  = config['training']['sample_weight_low']

    vol_thresh = np.abs(train_df['future_return']).median()
    weights = np.where(np.abs(train_df['future_return']) >= vol_thresh, w_high, w_low)

    logger.info(f"Sample weights: high={w_high}x, low={w_low}x, vol_threshold={vol_threshold:.5f}")
    return weights

def reduce_features_via_lgbm_v1(X_train, y_train, X_val, y_val, w_train, config):
    lgb_cfg = config['training']['lightgbm_screener']
    target_features = config['features']['target_n_features']

    # Remove non-feature params
    params = {k: v for k, v in lgb_cfg.items() if k not in ['early_stopping_rounds']}
    
    screener = lgb.LGBMClassifier(**params)
    screener.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(lgb_cfg['early_stopping_rounds'], verbose=False),
        ]
    )

    importance = pd.Series(screener.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    nonzero = importance[importance > 0]
    top_features = nonzero.head(target_features).index.tolist()

    logger.info(f"Feature reduction: {len(X_train.columns)} -> {len(top_features)}")
    return top_features

def optimize_catboost(X_train, y_train, X_val, y_val, w_train, config):
    cb_cfg = config['training']['catboost']
    opt_cfg = config['training']['optuna']
    
    def objective(trial):
        params = {
            'iterations': cb_cfg['iterations_per_trial'],
            'early_stopping_rounds': cb_cfg['early_stopping_per_trial'],
            'task_type': cb_cfg['task_type'],
            'devices': cb_cfg['devices'],
            'gpu_ram_part': cb_cfg['gpu_ram_part'],
            'max_bin': trial.suggest_int('max_bin', *cb_cfg['optuna_search_space']['max_bin']),
            'bootstrap_type': cb_cfg['bootstrap_type'],
            'loss_function': cb_cfg['loss_function'],
            'eval_metric': cb_cfg['eval_metric'],
            'random_seed': cb_cfg['random_seed'],
            'depth': trial.suggest_int('depth', *cb_cfg['optuna_search_space']['depth']),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *cb_cfg['optuna_search_space']['l2_leaf_reg'], log=True),
            'random_strength': trial.suggest_float('random_strength', *cb_cfg['optuna_search_space']['random_strength']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *cb_cfg['optuna_search_space']['min_data_in_leaf']),
            'learning_rate': trial.suggest_float('learning_rate', *cb_cfg['optuna_search_space']['learning_rate'], log=True),
            'subsample': trial.suggest_float('subsample', *cb_cfg['optuna_search_space']['subsample']),
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_val, y_val), use_best_model=True)
        
        preds_val = model.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, preds_val)
        
        # Overfit penalty
        preds_train = model.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, preds_train)
        gap = auc_train - auc_val
        penalty = max(0, gap - config['training']['optuna']['overfit_gap_allowed']) * 5.0
        
        return auc_val - penalty

    study = optuna.create_study(direction=opt_cfg['direction'], sampler=optuna.samplers.TPESampler(seed=opt_cfg['seed']))
    study.optimize(objective, n_trials=opt_cfg['n_trials'])
    
    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    return study.best_params

def train_final_models(X_train, y_train, X_val, y_val, w_train, best_params_cb, config):
    # 1. CatBoost Final
    cb_cfg = config['training']['catboost']
    cb_params = {
        'iterations': cb_cfg['iterations_final'],
        'early_stopping_rounds': cb_cfg['early_stopping_final'],
        'task_type': cb_cfg['task_type'],
        'devices': cb_cfg['devices'],
        'gpu_ram_part': cb_cfg['gpu_ram_part'],
        'bootstrap_type': cb_cfg['bootstrap_type'],
        'loss_function': cb_cfg['loss_function'],
        'eval_metric': cb_cfg['eval_metric'],
        'random_seed': cb_cfg['random_seed'],
        **best_params_cb,
        'verbose': 500
    }
    
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_val, y_val), use_best_model=True)
    
    # 2. LightGBM Final
    lgb_cfg = config['training']['lightgbm_final']
    lgb_model = lgb.LGBMClassifier(**{k: v for k, v in lgb_cfg.items() if k not in ['num_boost_round', 'early_stopping_rounds']})
    lgb_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(lgb_cfg['early_stopping_rounds'], verbose=False)]
    )
    
    return cb_model, lgb_model

def calibrate_and_threshold(cb_model, lgb_model, X_val, y_val, config):
    p_cb = cb_model.predict_proba(X_val)[:, 1]
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    
    # Ensemble weights based on AUC
    auc_cb = roc_auc_score(y_val, p_cb)
    auc_lgb = roc_auc_score(y_val, p_lgb)
    
    w_cb = auc_cb**2 / (auc_cb**2 + auc_lgb**2)
    w_lgb = 1 - w_cb
    
    p_ens = p_cb * w_cb + p_lgb * w_lgb
    
    # Calibrator
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_ens, y_val)
    
    # Threshold search
    p_cal = calibrator.predict(p_ens)
    best_acc, best_t = 0, 0.50
    scan_cfg = config['evaluation']
    
    for t in np.arange(scan_cfg['threshold_scan_start'], scan_cfg['threshold_scan_end'], scan_cfg['threshold_scan_step']):
        mask = (p_cal >= t) | (p_cal <= 1 - t)
        if mask.mean() < scan_cfg['min_coverage_for_threshold']: break
        
        acc = accuracy_score(y_val[mask], (p_cal[mask] >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
            
    logger.info(f"Optimal threshold: {best_t:.3f} | Accuracy: {best_acc:.2%}")
    return calibrator, (w_cb, w_lgb), best_t

def run_training_pipeline(config):
    # 1. Load data
    features_rel = Path(config['paths']['data']['features_v1'])
    features_path = PROJECT_ROOT / features_rel / "final_features.parquet"
    df = load_and_validate_data_v1(features_path, config)
    
    # 2. Filter features
    feature_cols = [c for c in df.columns if is_valid_feature(c, config)]
    
    # Filter only numeric columns and by NaN/const
    valid_cols = []
    for c in feature_cols:
        col_data = df[c]
        if not np.issubdtype(col_data.dtype, np.number):
            continue
        if col_data.isna().mean() > config['features']['nan_threshold']:
            continue
        if col_data.std() < config['features']['std_threshold']:
            continue
        valid_cols.append(c)
    
    logger.info(f"Valid features: {len(valid_cols)} / {len(feature_cols)}")
    
    # 3. Time Split
    df = df.sort_values('round_start')
    n = len(df)
    train_end = int(n * config['training']['train_ratio'])
    val_end = train_end + int(n * config['training']['val_ratio'])
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train, y_train = train_df[valid_cols], train_df['target']
    X_val, y_val = val_df[valid_cols], val_df['target']
    X_test, y_test = test_df[valid_cols], test_df['target']
    
    w_train = compute_sample_weights(train_df, config)
    
    # 4. Regime Detector
    regime_detector = RegimeDetector().fit(X_train)
    artifacts_rel = Path(config['paths']['models']['artifacts'])
    artifacts_path = PROJECT_ROOT / artifacts_rel
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    regime_detector.save(artifacts_path / "regime_detector_v1.pkl")
    
    # 5. Feature Selection
    selected_features = reduce_features_via_lgbm_v1(X_train, y_train, X_val, y_val, w_train, config)
    X_train_sub = X_train[selected_features].astype(np.float32)
    X_val_sub = X_val[selected_features].astype(np.float32)
    X_test_sub = X_test[selected_features].astype(np.float32)
    
    joblib.dump(selected_features, artifacts_path / "selected_features_v1.pkl")
    
    # 6. Optimization
    best_params_cb = optimize_catboost(X_train_sub, y_train, X_val_sub, y_val, w_train, config)
    joblib.dump(best_params_cb, artifacts_path / "best_params_v1.pkl")
    
    # 7. Final Training
    cb_model, lgb_model = train_final_models(X_train_sub, y_train, X_val_sub, y_val, w_train, best_params_cb, config)
    cb_model.save_model(str(artifacts_path / "catboost_final_v1.cbm"))
    joblib.dump(lgb_model, artifacts_path / "lgbm_final_v1.pkl")
    
    # 8. Calibration & Threshold
    calibrator, ensemble_weights, best_t = calibrate_and_threshold(cb_model, lgb_model, X_val_sub, y_val, config)
    joblib.dump(calibrator, artifacts_path / "calibrator_v1.pkl")
    joblib.dump(ensemble_weights, artifacts_path / "ensemble_weights_v1.pkl")
    joblib.dump(best_t, artifacts_path / "optimal_threshold_v1.pkl")
    
    # 9. Test Evaluation
    p_cb_test = cb_model.predict_proba(X_test_sub)[:, 1]
    p_lgb_test = lgb_model.predict_proba(X_test_sub)[:, 1]
    p_ens_test = p_cb_test * ensemble_weights[0] + p_lgb_test * ensemble_weights[1]
    p_cal_test = calibrator.predict(p_ens_test)
    
    auc_test = roc_auc_score(y_test, p_cal_test)
    mask = (p_cal_test >= best_t) | (p_cal_test <= 1 - best_t)
    acc_test = accuracy_score(y_test[mask], (p_cal_test[mask] >= best_t).astype(int)) if mask.any() else 0
    cov_test = mask.mean()
    
    logger.info(f"FINAL TEST RESULTS: AUC={auc_test:.4f} | Acc={acc_test:.2%} | Cov={cov_test:.2%}")

if __name__ == "__main__":
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_training_pipeline(config)
