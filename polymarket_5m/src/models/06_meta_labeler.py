import logging
from pathlib import Path

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_THRESHOLD = 0.530
MIN_TEST_COVERAGE = 0.50
MIN_SELECTION_COVERAGE = 0.54


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_artifacts(config: dict):
    artifacts_path = PROJECT_ROOT / Path(config["paths"]["models"]["artifacts"])

    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(artifacts_path / "catboost_final_v1.cbm"))

    return {
        "artifacts_path": artifacts_path,
        "cb_model": cb_model,
        "lgb_model": joblib.load(artifacts_path / "lgbm_final_v1.pkl"),
        "calibrator": joblib.load(artifacts_path / "calibrator_v1.pkl"),
        "ensemble_weights": joblib.load(artifacts_path / "ensemble_weights_v1.pkl"),
        "selected_features": joblib.load(artifacts_path / "selected_features_v1.pkl"),
    }


def load_feature_frame(config: dict) -> pd.DataFrame:
    features_path = PROJECT_ROOT / Path(config["paths"]["data"]["features_v1"]) / "final_features.parquet"
    df = pd.read_parquet(features_path)
    if "round_start" not in df.columns:
        df = df.reset_index()
    return df.sort_values("round_start").reset_index(drop=True)


def split_frame(df: pd.DataFrame, config: dict):
    n = len(df)
    train_end = int(n * config["training"]["train_ratio"])
    val_end = train_end + int(n * config["training"]["val_ratio"])
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def primary_prob(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    features = artifacts["selected_features"]
    X = df[features].astype(np.float32)
    p_cb = artifacts["cb_model"].predict_proba(X)[:, 1]
    p_lgb = artifacts["lgb_model"].predict_proba(X)[:, 1]
    w_cb, w_lgb = artifacts["ensemble_weights"]
    p_raw = p_cb * w_cb + p_lgb * w_lgb
    p_cal = artifacts["calibrator"].predict(p_raw)

    out = pd.DataFrame(index=df.index)
    out["p_cb"] = p_cb
    out["p_lgb"] = p_lgb
    out["p_raw"] = p_raw
    out["p_cal"] = p_cal
    out["primary_pred"] = (p_cal >= PRIMARY_THRESHOLD).astype(np.int8)
    out["primary_accept"] = (p_cal >= PRIMARY_THRESHOLD) | (p_cal <= 1.0 - PRIMARY_THRESHOLD)
    out["primary_conf"] = np.abs(p_cal - 0.5)
    out["primary_entropy"] = -(p_cal * np.log(np.clip(p_cal, 1e-6, 1.0)) + (1 - p_cal) * np.log(np.clip(1 - p_cal, 1e-6, 1.0)))
    out["model_disagreement"] = np.abs(p_cb - p_lgb)
    return out


def pick_optional_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "win_rvol_300",
        "snap_rvol_300",
        "win_mean_rvol_300",
        "win_mean_spread_bps",
        "snap_spread_bps",
        "win_mean_obi",
        "snap_obi",
        "win_mean_ofi",
        "snap_ofi",
        "win_mean_vpin",
        "snap_vpin",
        "win_mean_trade_imbalance",
        "snap_trade_imbalance",
        "win_ret_1",
        "win_ret_3",
        "win_ret_5",
        "win_mean_log_return",
        "win_std_log_return",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "is_us_session",
        "is_london_session",
        "is_overlap",
        "is_weekend",
        "minutes_since_roll",
    ]
    return [c for c in preferred if c in df.columns and np.issubdtype(df[c].dtype, np.number)]


def build_meta_features(df: pd.DataFrame, primary: pd.DataFrame, optional_cols: list[str]) -> pd.DataFrame:
    meta = primary[["p_cb", "p_lgb", "p_raw", "p_cal", "primary_conf", "primary_entropy", "model_disagreement"]].copy()
    if optional_cols:
        extra = df[optional_cols].replace([np.inf, -np.inf], np.nan)
        meta = pd.concat([meta, extra], axis=1)
    return meta.astype(np.float32)


def evaluate_primary(name: str, y: pd.Series, primary: pd.DataFrame) -> dict:
    mask = primary["primary_accept"].to_numpy()
    pred = primary["primary_pred"].to_numpy()
    acc = accuracy_score(y[mask], pred[mask]) if mask.any() else np.nan
    cov = float(mask.mean())
    auc = roc_auc_score(y, primary["p_cal"])
    logger.info("%s primary @ %.3f: AUC=%.4f | Acc=%.2f%% | Cov=%.2f%% (%d)",
                name, PRIMARY_THRESHOLD, auc, acc * 100, cov * 100, int(mask.sum()))
    return {"auc": auc, "acc": acc, "coverage": cov, "n": int(mask.sum())}


def choose_meta_threshold(y: pd.Series, primary: pd.DataFrame, q: np.ndarray, min_coverage: float) -> tuple[float, dict]:
    primary_mask = primary["primary_accept"].to_numpy()
    primary_pred = primary["primary_pred"].to_numpy()

    candidates = np.unique(np.quantile(q[primary_mask], np.linspace(0.0, 0.95, 96)))
    best = {"q_threshold": 0.0, "acc": -np.inf, "coverage": 0.0, "n": 0}

    for q_threshold in candidates:
        final_mask = primary_mask & (q >= q_threshold)
        coverage = float(final_mask.mean())
        if coverage < min_coverage or final_mask.sum() < 30:
            continue

        acc = accuracy_score(y[final_mask], primary_pred[final_mask])
        if acc > best["acc"] or (np.isclose(acc, best["acc"]) and coverage > best["coverage"]):
            best = {
                "q_threshold": float(q_threshold),
                "acc": float(acc),
                "coverage": coverage,
                "n": int(final_mask.sum()),
            }

    if best["acc"] == -np.inf:
        q_threshold = float(np.min(q[primary_mask])) if primary_mask.any() else 0.0
        final_mask = primary_mask & (q >= q_threshold)
        best = {
            "q_threshold": q_threshold,
            "acc": float(accuracy_score(y[final_mask], primary_pred[final_mask])) if final_mask.any() else np.nan,
            "coverage": float(final_mask.mean()),
            "n": int(final_mask.sum()),
        }

    return best["q_threshold"], best


def evaluate_meta(name: str, y: pd.Series, primary: pd.DataFrame, q: np.ndarray, q_threshold: float) -> dict:
    final_mask = primary["primary_accept"].to_numpy() & (q >= q_threshold)
    pred = primary["primary_pred"].to_numpy()
    acc = accuracy_score(y[final_mask], pred[final_mask]) if final_mask.any() else np.nan
    cov = float(final_mask.mean())
    logger.info("%s meta @ primary %.3f + q>=%.4f: Acc=%.2f%% | Cov=%.2f%% (%d)",
                name, PRIMARY_THRESHOLD, q_threshold, acc * 100, cov * 100, int(final_mask.sum()))
    return {"acc": float(acc), "coverage": cov, "n": int(final_mask.sum())}


def scaled_lgb_prob(df: pd.DataFrame, artifacts: dict, scale: float) -> np.ndarray:
    X = df[artifacts["selected_features"]].astype(np.float32)
    p_lgb = artifacts["lgb_model"].predict_proba(X)[:, 1]
    return np.clip(0.5 + scale * (p_lgb - 0.5), 0.0, 1.0)


def evaluate_threshold_policy(name: str, y: pd.Series, p: np.ndarray) -> dict:
    mask = (p >= PRIMARY_THRESHOLD) | (p <= 1.0 - PRIMARY_THRESHOLD)
    pred = (p >= PRIMARY_THRESHOLD).astype(np.int8)
    acc = accuracy_score(y[mask], pred[mask]) if mask.any() else np.nan
    cov = float(mask.mean())
    logger.info("%s threshold policy @ %.3f: Acc=%.2f%% | Cov=%.2f%% (%d)",
                name, PRIMARY_THRESHOLD, acc * 100, cov * 100, int(mask.sum()))
    return {"acc": float(acc), "coverage": cov, "n": int(mask.sum())}


def choose_lgb_scale(selection_df: pd.DataFrame, y_selection: pd.Series, artifacts: dict) -> tuple[float, dict]:
    best = {"scale": 1.0, "acc": -np.inf, "coverage": 0.0, "n": 0}

    for scale in np.linspace(1.0, 2.0, 41):
        p = scaled_lgb_prob(selection_df, artifacts, scale)
        mask = (p >= PRIMARY_THRESHOLD) | (p <= 1.0 - PRIMARY_THRESHOLD)
        pred = (p >= PRIMARY_THRESHOLD).astype(np.int8)
        metrics = {
            "acc": float(accuracy_score(y_selection[mask], pred[mask])) if mask.any() else np.nan,
            "coverage": float(mask.mean()),
            "n": int(mask.sum()),
        }
        if metrics["coverage"] < MIN_SELECTION_COVERAGE:
            continue
        if metrics["acc"] > best["acc"] or (np.isclose(metrics["acc"], best["acc"]) and metrics["coverage"] > best["coverage"]):
            best = {"scale": float(scale), **metrics}

    if best["acc"] == -np.inf:
        scale = 1.0
        p = scaled_lgb_prob(selection_df, artifacts, scale)
        best = {"scale": scale, **evaluate_threshold_policy("SCALE_FALLBACK", y_selection, p)}

    return best["scale"], best


def run_meta_labeling():
    config = load_config()
    artifacts = load_artifacts(config)
    df = load_feature_frame(config)
    _, val_df, test_df = split_frame(df, config)

    val_primary = primary_prob(val_df, artifacts)
    test_primary = primary_prob(test_df, artifacts)

    y_val = val_df["target"].astype(np.int8).reset_index(drop=True)
    y_test = test_df["target"].astype(np.int8).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_primary = val_primary.reset_index(drop=True)
    test_primary = test_primary.reset_index(drop=True)

    evaluate_primary("VAL", y_val, val_primary)
    test_primary_metrics = evaluate_primary("TEST", y_test, test_primary)

    split_at = int(len(val_df) * 0.60)
    meta_train_df = val_df.iloc[:split_at].reset_index(drop=True)
    meta_select_df = val_df.iloc[split_at:].reset_index(drop=True)
    meta_train_primary = val_primary.iloc[:split_at].reset_index(drop=True)
    meta_select_primary = val_primary.iloc[split_at:].reset_index(drop=True)
    y_meta_train = y_val.iloc[:split_at].reset_index(drop=True)
    y_meta_select = y_val.iloc[split_at:].reset_index(drop=True)

    train_accept = meta_train_primary["primary_accept"].to_numpy()
    optional_cols = pick_optional_feature_columns(df)
    X_meta_train = build_meta_features(meta_train_df, meta_train_primary, optional_cols)
    X_meta_select = build_meta_features(meta_select_df, meta_select_primary, optional_cols)
    X_test_meta = build_meta_features(test_df, test_primary, optional_cols)

    meta_target = (meta_train_primary["primary_pred"].to_numpy() == y_meta_train.to_numpy()).astype(np.int8)
    meta_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=7,
        min_child_samples=80,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=10,
        verbose=-1,
    )
    meta_model.fit(X_meta_train.loc[train_accept], meta_target[train_accept])

    q_select = meta_model.predict_proba(X_meta_select)[:, 1]
    q_test = meta_model.predict_proba(X_test_meta)[:, 1]

    q_threshold, select_metrics = choose_meta_threshold(
        y_meta_select,
        meta_select_primary,
        q_select,
        min_coverage=MIN_TEST_COVERAGE,
    )
    logger.info(
        "META threshold selected on late VAL: q>=%.4f | Acc=%.2f%% | Cov=%.2f%% (%d)",
        q_threshold,
        select_metrics["acc"] * 100,
        select_metrics["coverage"] * 100,
        select_metrics["n"],
    )

    test_meta_metrics = evaluate_meta("TEST", y_test, test_primary, q_test, q_threshold)

    logger.info("\n--- LGB confidence scaling policy ---")
    lgb_scale, lgb_select_metrics = choose_lgb_scale(meta_select_df, y_meta_select, artifacts)
    logger.info(
        "LGB scale selected on late VAL: scale=%.3f | Acc=%.2f%% | Cov=%.2f%% (%d)",
        lgb_scale,
        lgb_select_metrics["acc"] * 100,
        lgb_select_metrics["coverage"] * 100,
        lgb_select_metrics["n"],
    )
    p_lgb_scaled_test = scaled_lgb_prob(test_df, artifacts, lgb_scale)
    test_lgb_scaled_metrics = evaluate_threshold_policy("TEST LGB-scaled", y_test, p_lgb_scaled_test)

    artifacts_path = artifacts["artifacts_path"]
    joblib.dump(meta_model, artifacts_path / "meta_labeler_v1.pkl")
    joblib.dump(list(X_test_meta.columns), artifacts_path / "meta_features_v1.pkl")
    joblib.dump(
        {
            "primary_threshold": PRIMARY_THRESHOLD,
            "meta_threshold": q_threshold,
            "min_coverage": MIN_TEST_COVERAGE,
            "optional_feature_cols": optional_cols,
            "selection_metrics": select_metrics,
            "test_primary_metrics": test_primary_metrics,
            "test_meta_metrics": test_meta_metrics,
            "lgb_scale": lgb_scale,
            "lgb_selection_metrics": lgb_select_metrics,
            "test_lgb_scaled_metrics": test_lgb_scaled_metrics,
        },
        artifacts_path / "meta_policy_v1.pkl",
    )
    joblib.dump(
        {
            "model": "lgbm_final_v1",
            "probability_transform": "clip(0.5 + scale * (p_lgb - 0.5), 0, 1)",
            "scale": lgb_scale,
            "threshold": PRIMARY_THRESHOLD,
            "selection_min_coverage": MIN_SELECTION_COVERAGE,
            "selection_metrics": lgb_select_metrics,
            "test_metrics": test_lgb_scaled_metrics,
        },
        artifacts_path / "threshold_policy_v1.pkl",
    )

    delta_acc = test_meta_metrics["acc"] - test_primary_metrics["acc"]
    delta_cov = test_meta_metrics["coverage"] - test_primary_metrics["coverage"]
    logger.info("TEST delta vs primary: Acc=%+.2f pp | Cov=%+.2f pp", delta_acc * 100, delta_cov * 100)
    lgb_delta_acc = test_lgb_scaled_metrics["acc"] - test_primary_metrics["acc"]
    lgb_delta_cov = test_lgb_scaled_metrics["coverage"] - test_primary_metrics["coverage"]
    logger.info("TEST LGB-scaled delta vs primary: Acc=%+.2f pp | Cov=%+.2f pp", lgb_delta_acc * 100, lgb_delta_cov * 100)

    if test_lgb_scaled_metrics["acc"] >= 0.60 and test_lgb_scaled_metrics["coverage"] >= MIN_TEST_COVERAGE:
        logger.info("TARGET HIT: accuracy >= 60%% with coverage >= 50%%")
    else:
        logger.info("TARGET NOT HIT: keep this as a measured experiment, not production policy yet")


if __name__ == "__main__":
    run_meta_labeling()
