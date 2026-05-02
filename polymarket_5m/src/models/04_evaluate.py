import pandas as pd
import numpy as np
import logging
import yaml
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically find project root (assuming script is in src/models/04_evaluate.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def full_evaluation_v1(cb_model, lgb_model, calibrator, 
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        df_test_meta, ensemble_weights, feature_cols, config, best_t):
    """
    Laporan lengkap yang WAJIB ada sebelum model dianggap valid.
    """
    w_cb, w_lgb = ensemble_weights

    def ensemble_prob(X):
        p_cb  = cb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_raw = p_cb * w_cb + p_lgb * w_lgb
        return calibrator.predict(p_raw)  # Setelah kalibrasi

    p_train = ensemble_prob(X_train)
    p_val   = ensemble_prob(X_val)
    p_test  = ensemble_prob(X_test)

    # 1. AUC semua split
    auc_train = roc_auc_score(y_train, p_train)
    auc_val   = roc_auc_score(y_val,   p_val)
    auc_test  = roc_auc_score(y_test,  p_test)
    gap       = auc_train - auc_test

    logger.info(f"\nAUC — Train: {auc_train:.4f} | Val: {auc_val:.4f} | Test: {auc_test:.4f}")
    logger.info(f"Overfit gap: {gap:.4f} {'✅' if gap < 0.05 else '⚠️ OVERFIT'}")

    # 2. DISTRIBUSI PROBABILITAS
    logger.info("\n--- Distribusi Probabilitas (TEST) ---")
    bins = [(0.30,0.40), (0.40,0.45), (0.45,0.50), (0.50,0.55), (0.55,0.60), (0.60,0.70)]
    for lo, hi in bins:
        n = ((p_test >= lo) & (p_test < hi)).sum()
        logger.info(f"  [{lo:.2f}-{hi:.2f}): {n:4d} ({n/len(p_test)*100:.1f}%)")

    flat_pct = ((p_test >= 0.45) & (p_test < 0.55)).mean()
    if flat_pct > 0.80:
        logger.warning(f"⚠️ {flat_pct:.1%} prob berada di [0.45, 0.55] — model hampir tidak ada signal")

    # 3. Accuracy di TEST dengan berbagai threshold
    logger.info("\n--- Accuracy di TEST dengan berbagai threshold ---")
    thresholds = config['evaluation']['confidence_levels'] + [best_t]
    thresholds = sorted(list(set(thresholds)))
    
    for t in thresholds:
        mask = (p_test >= t) | (p_test <= 1 - t)
        n    = mask.sum()
        if n < 30: continue
        acc  = accuracy_score(y_test[mask], (p_test[mask] >= t).astype(int))
        cov  = n / len(p_test)
        logger.info(f"  t={t:.3f}: Acc={acc:.2%} | Cov={cov:.2%} ({n} rounds)")

    # 4. FEATURE IMPORTANCE BY CATEGORY
    imp = pd.DataFrame({'feature': feature_cols,
                        'importance': cb_model.get_feature_importance()
                       }).sort_values('importance', ascending=False)

    category_map = {
        'snap_'   : 'Microstructure Snapshot',
        'win_mean_': 'Window Mean',
        'win_std_' : 'Window Std',
        'win_trend_': 'Window Trend',
        'win_ofi' : 'OFI Accel/Window',
        'past_'   : 'Inter-round Lag',
        'win_ret' : 'Price Action',
    }
    logger.info("\n--- Feature Category Importance ---")
    for prefix, name in category_map.items():
        cat_feats = [f for f in imp['feature'] if f.startswith(prefix)]
        if cat_feats:
            total_imp = imp[imp['feature'].isin(cat_feats)]['importance'].sum()
            logger.info(f"  {name:<30}: {total_imp:.1f}")

    # 5. PER-SESSION ANALYSIS
    # Pastikan round_start ada dan ambil jam-nya secara aman
    if 'round_start' in df_test_meta.columns:
        hours = pd.to_datetime(df_test_meta['round_start']).dt.hour
    else:
        hours = pd.to_datetime(df_test_meta.index).hour
    
    sessions = {
        'US (13-21 UTC)'     : (hours >= 13) & (hours < 21),
        'London (07-16 UTC)' : (hours >= 7)  & (hours < 16),
        'Asian (00-08 UTC)'  : (hours >= 0)  & (hours < 8),
        'Overlap (13-16 UTC)': (hours >= 13) & (hours < 16),
    }
    logger.info("\n--- Per-session Accuracy (threshold 0.5) ---")
    for name, mask in sessions.items():
        if mask.sum() > 50:
            acc = accuracy_score(y_test[mask], (p_test[mask] >= 0.5).astype(int))
            logger.info(f"  {name}: {acc:.2%} ({mask.sum()} rounds)")

    # 6. PER-VOLATILITY-REGIME ANALYSIS
    if 'win_rvol_300' in df_test_meta.columns:
        rvol = df_test_meta['win_rvol_300']
        low_vol  = rvol < rvol.quantile(config['evaluation']['rvol_low_pct'])
        high_vol = rvol > rvol.quantile(config['evaluation']['rvol_high_pct'])
        normal   = ~low_vol & ~high_vol

        logger.info("\n--- Per-volatility-regime Accuracy ---")
        for name, mask in [('Low Vol', low_vol), ('Normal Vol', normal), ('High Vol', high_vol)]:
            if mask.sum() > 50:
                acc = accuracy_score(y_test[mask], (p_test[mask] >= best_t).astype(int))
                logger.info(f"  {name}: {acc:.2%} ({mask.sum()} rounds)")

def walk_forward_validation_v1(df: pd.DataFrame, feature_cols: list, config: dict):
    """
    Single split tidak cukup. Walk-forward memberikan estimasi yang lebih andal.
    """
    wfv_cfg       = config['training']
    train_rounds  = wfv_cfg['wfv_train_rounds']
    val_rounds    = wfv_cfg['wfv_val_rounds']
    step_rounds   = wfv_cfg['wfv_step_rounds']
    n_splits      = wfv_cfg['wfv_n_splits']

    df    = df.sort_values('round_start').reset_index(drop=True)
    n     = len(df)
    results = []

    for i in range(n_splits):
        val_end   = n - (n_splits - 1 - i) * step_rounds
        val_start = val_end - val_rounds
        train_end = val_start
        train_start = max(0, train_end - train_rounds)

        if val_start < train_rounds or val_end > n:
            continue

        X_tr = df.iloc[train_start:train_end][feature_cols].astype(np.float32)
        y_tr = df.iloc[train_start:train_end]['target'].astype(np.int8)
        X_va = df.iloc[val_start:val_end][feature_cols].astype(np.float32)
        y_va = df.iloc[val_start:val_end]['target'].astype(np.int8)

        # Quick training
        model = cb.CatBoostClassifier(
            iterations=1000, depth=4, learning_rate=0.01,
            task_type='GPU', gpu_ram_part=0.40,
            bootstrap_type='Bernoulli', loss_function='Logloss',
            eval_metric='AUC', early_stopping_rounds=50,
            verbose=False, random_seed=42
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

        p_va = model.predict_proba(X_va)[:, 1]
        auc  = roc_auc_score(y_va, p_va)
        acc  = accuracy_score(y_va, (p_va >= 0.5).astype(int))

        period = pd.to_datetime(df.iloc[val_start]['round_start'])
        results.append({'fold': i+1, 'period': period, 'auc': auc, 'accuracy': acc})
        logger.info(f"WFV Fold {i+1}: {period.date()} | AUC={auc:.4f} | Acc={acc:.2%}")

    results_df = pd.DataFrame(results)
    logger.info(f"\nWFV Summary: AUC mean={results_df['auc'].mean():.4f} std={results_df['auc'].std():.4f}")
    return results_df

if __name__ == "__main__":
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    artifacts_rel = Path(config['paths']['models']['artifacts'])
    artifacts_path = PROJECT_ROOT / artifacts_rel
    
    # Load artifacts
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(artifacts_path / "catboost_final_v1.cbm"))
    lgb_model = joblib.load(artifacts_path / "lgbm_final_v1.pkl")
    calibrator = joblib.load(artifacts_path / "calibrator_v1.pkl")
    ensemble_weights = joblib.load(artifacts_path / "ensemble_weights_v1.pkl")
    best_t = joblib.load(artifacts_path / "optimal_threshold_v1.pkl")
    selected_features = joblib.load(artifacts_path / "selected_features_v1.pkl")

    # Load data for evaluation
    features_rel = Path(config['paths']['data']['features_v1'])
    features_path = PROJECT_ROOT / features_rel / "final_features.parquet"
    df = pd.read_parquet(features_path)

    if 'round_start' not in df.columns:
        df = df.reset_index()

    df = df.sort_values('round_start')
    
    n = len(df)
    train_end = int(n * config['training']['train_ratio'])
    val_end = train_end + int(n * config['training']['val_ratio'])
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train = train_df[selected_features].astype(np.float32)
    y_train = train_df['target']
    X_val = val_df[selected_features].astype(np.float32)
    y_val = val_df['target']
    X_test = test_df[selected_features].astype(np.float32)
    y_test = test_df['target']
    
    full_evaluation_v1(cb_model, lgb_model, calibrator, 
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        test_df, ensemble_weights, selected_features, config, best_t)

    # 10. WALK-FORWARD VALIDATION (Dukungan penuh arsitektur)
    logger.info("\n--- Memulai Walk-Forward Validation (6-folds) ---")
    walk_forward_validation_v1(df, selected_features, config)
