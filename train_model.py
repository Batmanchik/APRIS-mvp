from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from data_generator import FEATURE_COLUMNS, RISK_THRESHOLDS, SEED, build_dataset, validate_dataset  # type: ignore


ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = ARTIFACTS_DIR / "synthetic_dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
IMPORTANCES_JSON_PATH = ARTIFACTS_DIR / "feature_importances.json"
ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.png"


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_divide(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _prepare_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        validate_dataset(df)
        return df

    df = build_dataset(total_n=4000, seed=SEED)
    df.to_csv(DATASET_PATH, index=False)
    return df


def _check_no_data_leakage(x: pd.DataFrame) -> None:
    forbidden = {"label", "is_borderline"}
    leak_cols = [col for col in x.columns if col in forbidden]
    assert not leak_cols, f"Data leakage risk: forbidden columns in features: {leak_cols}"


def _check_nan(x: pd.DataFrame, y: pd.Series) -> None:
    assert not x.isna().any().any(), "NaN found in feature matrix."
    assert not y.isna().any(), "NaN found in target vector."
    assert np.isfinite(x.to_numpy(dtype=float)).all(), "Inf found in feature matrix."


def _plot_and_save_roc(y_true: pd.Series, y_score: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=140)
    plt.close(fig)
    return {
        "fpr": [float(v) for v in fpr.tolist()],
        "tpr": [float(v) for v in tpr.tolist()],
    }


def train_and_save(df: pd.DataFrame, random_state: int = SEED) -> dict[str, Any]:
    x = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()

    _check_no_data_leakage(x)
    _check_nan(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )
    assert set(x_train.index).isdisjoint(set(x_test.index)), "Data leakage: train/test index overlap."

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    roc_points = _plot_and_save_roc(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()
    recall_pyramid = _safe_divide(tp, tp + fn)
    precision_pyramid = _safe_divide(tp, tp + fp)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "recall_pyramid": recall_pyramid,
        "precision_pyramid": precision_pyramid,
        "confusion_matrix": cm.tolist(),
        "threshold_policy": "fixed_0.4_0.7",
        "threshold_values": dict(RISK_THRESHOLDS),
        "random_state": random_state,
        "dataset_rows": int(len(df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "x_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "y_train_shape": [int(y_train.shape[0])],
        "model": {
            "name": "RandomForestClassifier",
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "random_state": random_state,
                "n_jobs": -1,
            },
        },
        "roc_curve_path": str(ROC_CURVE_PATH),
    }

    _ensure_artifacts_dir()
    joblib.dump(model, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importances_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values("importance", ascending=False)
    importances = [
        {"feature": str(row["feature"]), "importance": float(row["importance"])}
        for _, row in importances_df.iterrows()
    ]
    IMPORTANCES_JSON_PATH.write_text(json.dumps(importances, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "importances": importances,
        "roc_points": roc_points,
    }


def main() -> None:
    dataset = _prepare_dataset()
    result = train_and_save(dataset, random_state=SEED)
    metrics = result["metrics"]
    importances = result["importances"]
    roc_points = result["roc_points"]

    print("Training: OK")
    print(f"X_train shape: {tuple(metrics['x_train_shape'])}")
    print(f"y_train shape: {tuple(metrics['y_train_shape'])}")
    print("Leakage check: PASSED")
    print("NaN/Inf check: PASSED")
    print()
    print("Metrics:")
    print(f"recall_pyramid: {metrics['recall_pyramid']:.6f}")
    print(f"precision_pyramid: {metrics['precision_pyramid']:.6f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    print(f"roc_auc: {metrics['roc_auc']:.6f}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print()
    print("Top-5 feature importances:")
    for item in importances[:5]:
        print(f"  {item['feature']}: {item['importance']:.6f}")
    print()
    print("ROC curve points (sample):")
    sample_points = min(8, len(roc_points["fpr"]))
    for idx in range(sample_points):
        print(f"  fpr={roc_points['fpr'][idx]:.6f}, tpr={roc_points['tpr'][idx]:.6f}")
    print(f"roc_curve_saved: {ROC_CURVE_PATH}")
    print()
    print(json.dumps(metrics, indent=2))
    print(f"saved_model: {MODEL_PATH}")
    print(f"saved_features: {FEATURE_NAMES_PATH}")
    print(f"saved_metrics: {METRICS_PATH}")
    print(f"saved_feature_importances: {IMPORTANCES_JSON_PATH}")


if __name__ == "__main__":
    main()
