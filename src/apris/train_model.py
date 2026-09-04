from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from apris.cheops.infrastructure.ml.benchmark_v2 import (
    run_tabular_benchmark,
    save_benchmark_report,
)
from apris.cheops.infrastructure.ml.drift_v2 import (
    FEATURE_PROFILE_V2_PATH,
    build_drift_report,
    build_feature_profile,
    save_drift_report,
    save_feature_profile,
)
from apris.cheops.infrastructure.ml.fusion_v2 import (
    FUSION_V2_ARTIFACT_PATH,
    FUSION_V2_METRICS_PATH,
    save_fusion_artifact,
    save_fusion_metrics,
    train_fusion_meta,
)
from apris.cheops.infrastructure.ml.sequence_v2 import (
    SEQUENCE_V2_ARTIFACT_PATH,
    SEQUENCE_V2_METRICS_PATH,
    save_sequence_artifact,
    save_sequence_metrics,
    train_sequence_artifact,
)
from apris.cheops.infrastructure.ml.graph_v2 import (
    GRAPH_V2_ARTIFACT_PATH,
    GRAPH_V2_METRICS_PATH,
    save_graph_artifact,
    save_graph_metrics,
    train_graph_artifact,
)
from apris.cheops.infrastructure.ml.tabular_v2 import (
    TABULAR_V2_BUNDLE_PATH,
    TABULAR_V2_METRICS_PATH,
    save_tabular_bundle,
    save_tabular_metrics,
    train_tabular_bundle,
)
from apris.cheops.infrastructure.ml.model_registry_v2 import (
    MODEL_REGISTRY_V2_PATH,
    attach_benchmark_report,
    build_model_registry,
    load_model_registry,
    save_model_registry,
)
from apris.data_generator import (
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    RISK_THRESHOLDS,
    SEED,
    build_dataset,
    validate_dataset,
)
from apris.etl import process_external_dataset

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


def _validate_training_dataset(df: pd.DataFrame, *, source: str) -> None:
    required = set(FEATURE_COLUMNS + ["label"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")

    x = df[FEATURE_COLUMNS].astype(float)
    if x.isna().any().any():
        raise ValueError(f"{source}: NaN found in feature matrix.")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: Inf found in feature matrix.")

    y_raw = pd.to_numeric(df["label"], errors="coerce")
    if y_raw.isna().any():
        raise ValueError(f"{source}: label contains non-numeric values.")
    y_values = y_raw.to_numpy(dtype=float)
    if not np.isin(y_values, [0.0, 1.0]).all():
        raise ValueError(f"{source}: label must contain only 0/1 values.")
    if len(np.unique(y_values)) < 2:
        raise ValueError(f"{source}: label must contain at least two classes for stratified split.")

    for feature_name, (low, high) in FEATURE_BOUNDS.items():
        series = x[feature_name]
        if (series < low).any() or (series > high).any():
            raise ValueError(
                f"{source}: feature '{feature_name}' has values outside [{low}, {high}]"
            )


def _validate_dataset_for_source(df: pd.DataFrame, *, source: str) -> None:
    # Synthetic datasets include `is_borderline`; keep strict generation quality checks there.
    if "is_borderline" in df.columns:
        validate_dataset(df)
        return
    _validate_training_dataset(df, source=source)


def _prepare_dataset(external_data_path: str | None = None) -> pd.DataFrame:
    if external_data_path:
        print(f"Loading external dataset from {external_data_path} through ETL pipeline...")
        df = process_external_dataset(external_data_path)
        _validate_dataset_for_source(df, source=f"external dataset '{external_data_path}'")
        return df

    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        _validate_dataset_for_source(df, source=f"cached dataset '{DATASET_PATH}'")
        return df

    df = build_dataset(total_n=4000, seed=SEED)
    df.to_csv(DATASET_PATH, index=False)
    return df


def _load_features_for_drift(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Drift dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported drift dataset format: {path.suffix}. Expected .csv or .json")

    missing = [name for name in FEATURE_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(
            f"Drift dataset must contain model feature columns: missing {missing}"
        )
    features_df = df[FEATURE_COLUMNS].astype(float).copy()
    if features_df.isna().any().any():
        raise ValueError("Drift dataset contains NaN values in feature columns.")
    if not np.isfinite(features_df.to_numpy(dtype=float)).all():
        raise ValueError("Drift dataset contains non-finite values in feature columns.")
    return features_df


def _check_no_data_leakage(x: pd.DataFrame) -> None:
    forbidden = {"label", "is_borderline"}
    leak_cols = [col for col in x.columns if col in forbidden]
    if leak_cols:
        raise ValueError(f"Data leakage risk: forbidden columns in features: {leak_cols}")


def _check_nan(x: pd.DataFrame, y: pd.Series) -> None:
    if x.isna().any().any():
        raise ValueError("NaN found in feature matrix.")
    if y.isna().any():
        raise ValueError("NaN found in target vector.")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Inf found in feature matrix.")


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
    _ensure_artifacts_dir()
    _validate_training_dataset(df, source="training dataset")

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
    if not set(x_train.index).isdisjoint(set(x_test.index)):
        raise ValueError("Data leakage: train/test index overlap.")

    # MLflow tracking
    mlflow.set_experiment("CheopsAI_Risk_Model")
    with mlflow.start_run():
        model_params: dict[str, Any] = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_samples": 5,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        mlflow.log_params(model_params)

        model = lgb.LGBMClassifier(**model_params)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        roc_points = _plot_and_save_roc(y_test, y_proba)

        tn, fp, fn, tp = cm.ravel()
        recall_pyramid = _safe_divide(tp, tp + fn)
        precision_pyramid = _safe_divide(tp, tp + fp)

        accuracy = float(accuracy_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        mlflow.log_metrics({
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "recall_pyramid": recall_pyramid,
            "precision_pyramid": precision_pyramid,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
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
                "name": "LightGBMClassifier",
                "params": model_params,
            },
            "roc_curve_path": str(ROC_CURVE_PATH),
        }

        joblib.dump(model, MODEL_PATH)
        FEATURE_NAMES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        importances_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": np.asarray(model.feature_importances_, dtype=float),
            }
        ).sort_values("importance", ascending=False)
        
        # In LightGBM feature importances might be large integers (split counts) 
        # Normalize them to sum to 1.0 for consistency with previous Random Forest
        total_importance = importances_df["importance"].sum()
        if total_importance > 0:
            importances_df["importance"] = importances_df["importance"] / total_importance

        importances = [
            {"feature": str(row["feature"]), "importance": float(row["importance"])}
            for _, row in importances_df.iterrows()
        ]
        IMPORTANCES_JSON_PATH.write_text(json.dumps(importances, indent=2), encoding="utf-8")

        # Train Cheops v2 tabular branch (global + typology probabilities) with calibration.
        tabular_bundle, tabular_metrics = train_tabular_bundle(df, random_state=random_state)
        bundle_path = save_tabular_bundle(tabular_bundle)
        tabular_metrics_path = save_tabular_metrics(tabular_metrics)

        # Train Cheops v2 sequence branch (trainable temporal surrogate + calibration).
        sequence_artifact, sequence_metrics = train_sequence_artifact(
            df,
            random_state=random_state,
        )
        sequence_artifact_path = save_sequence_artifact(sequence_artifact)
        sequence_metrics_path = save_sequence_metrics(sequence_metrics)

        # Train Cheops v2 graph branch (trainable topology surrogate + calibration).
        graph_artifact, graph_metrics = train_graph_artifact(
            df,
            random_state=random_state,
        )
        graph_artifact_path = save_graph_artifact(graph_artifact)
        graph_metrics_path = save_graph_metrics(graph_metrics)

        # Train Cheops v2 fusion meta-head (tabular + sequence + graph).
        fusion_artifact, fusion_metrics = train_fusion_meta(
            df,
            tabular_bundle,
            random_state=random_state,
            sequence_artifact=sequence_artifact,
            graph_artifact=graph_artifact,
        )
        fusion_artifact_path = save_fusion_artifact(fusion_artifact)
        fusion_metrics_path = save_fusion_metrics(fusion_metrics)

        # Save baseline feature profile for future drift checks.
        feature_profile = build_feature_profile(
            x,
            dataset_name="train_dataset",
            random_state=random_state,
        )
        feature_profile_path = save_feature_profile(feature_profile)

        model_registry = build_model_registry(
            legacy_metrics=metrics,
            tabular_metrics=tabular_metrics,
            sequence_metrics=sequence_metrics,
            graph_metrics=graph_metrics,
            fusion_metrics=fusion_metrics,
            benchmark_report=None,
        )
        registry_path = save_model_registry(model_registry)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(FEATURE_NAMES_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(IMPORTANCES_JSON_PATH))
        mlflow.log_artifact(str(ROC_CURVE_PATH))
        mlflow.log_artifact(str(bundle_path))
        mlflow.log_artifact(str(tabular_metrics_path))
        mlflow.log_artifact(str(sequence_artifact_path))
        mlflow.log_artifact(str(sequence_metrics_path))
        mlflow.log_artifact(str(graph_artifact_path))
        mlflow.log_artifact(str(graph_metrics_path))
        mlflow.log_artifact(str(fusion_artifact_path))
        mlflow.log_artifact(str(fusion_metrics_path))
        mlflow.log_artifact(str(feature_profile_path))
        mlflow.log_artifact(str(registry_path))
        mlflow.sklearn.log_model(model, "lightgbm-model")

        return {
            "metrics": metrics,
            "importances": importances,
            "roc_points": roc_points,
            "tabular_v2_metrics": tabular_metrics,
            "sequence_v2_metrics": sequence_metrics,
            "graph_v2_metrics": graph_metrics,
            "fusion_v2_metrics": fusion_metrics,
            "feature_profile": feature_profile,
            "model_registry": model_registry,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cheops AI Risk Model")
    parser.add_argument("--data", type=str, help="Path to external CSV/JSON dataset (optional)")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run tabular benchmark (LightGBM + optional CatBoost/XGBoost) and save report.",
    )
    parser.add_argument(
        "--benchmark-lightgbm-only",
        action="store_true",
        help="Run benchmark only for LightGBM (skip optional CatBoost/XGBoost).",
    )
    parser.add_argument(
        "--drift-data",
        type=str,
        help="Optional CSV/JSON dataset with FEATURE_COLUMNS to compute drift report.",
    )
    args = parser.parse_args()

    dataset = _prepare_dataset(external_data_path=args.data)
    result = train_and_save(dataset, random_state=SEED)
    metrics = result["metrics"]
    importances = result["importances"]
    roc_points = result["roc_points"]

    print("Training: OK (with MLflow tracking)")
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
    print(f"saved_model: {MODEL_PATH}")
    print(f"saved_features: {FEATURE_NAMES_PATH}")
    print(f"saved_metrics: {METRICS_PATH}")
    print(f"saved_feature_importances: {IMPORTANCES_JSON_PATH}")
    print(f"saved_tabular_v2_bundle: {TABULAR_V2_BUNDLE_PATH}")
    print(f"saved_tabular_v2_metrics: {TABULAR_V2_METRICS_PATH}")
    print(f"saved_sequence_v2_artifact: {SEQUENCE_V2_ARTIFACT_PATH}")
    print(f"saved_sequence_v2_metrics: {SEQUENCE_V2_METRICS_PATH}")
    print(f"saved_graph_v2_artifact: {GRAPH_V2_ARTIFACT_PATH}")
    print(f"saved_graph_v2_metrics: {GRAPH_V2_METRICS_PATH}")
    print(f"saved_fusion_v2_artifact: {FUSION_V2_ARTIFACT_PATH}")
    print(f"saved_fusion_v2_metrics: {FUSION_V2_METRICS_PATH}")
    print(f"saved_feature_profile_v2: {FEATURE_PROFILE_V2_PATH}")
    print(f"saved_model_registry_v2: {MODEL_REGISTRY_V2_PATH}")

    if args.benchmark:
        benchmark_report = run_tabular_benchmark(
            dataset,
            random_state=SEED,
            include_optional=(not args.benchmark_lightgbm_only),
        )
        benchmark_path = save_benchmark_report(benchmark_report)

        try:
            registry_payload = load_model_registry()
        except FileNotFoundError:
            registry_payload = build_model_registry(
                legacy_metrics=result["metrics"],
                tabular_metrics=result["tabular_v2_metrics"],
                sequence_metrics=result["sequence_v2_metrics"],
                graph_metrics=result["graph_v2_metrics"],
                fusion_metrics=result["fusion_v2_metrics"],
                benchmark_report=benchmark_report,
            )
        else:
            registry_payload = attach_benchmark_report(registry_payload, benchmark_report)

        registry_path = save_model_registry(registry_payload)
        print()
        print("Tabular benchmark: OK")
        print(f"benchmark_winner: {benchmark_report['winner']}")
        print(f"saved_benchmark_report: {benchmark_path}")
        print(f"saved_model_registry_v2: {registry_path}")

    if args.drift_data:
        drift_features = _load_features_for_drift(args.drift_data)
        profile = result["feature_profile"]
        drift_report = build_drift_report(
            profile,
            drift_features,
            dataset_name=Path(args.drift_data).name,
        )
        drift_path = save_drift_report(drift_report)
        print()
        print("Drift check: OK")
        print(f"drift_overall_level: {drift_report['overall_level']}")
        print(f"drift_overall_psi: {drift_report['overall_psi']:.6f}")
        print(f"saved_drift_report: {drift_path}")


if __name__ == "__main__":
    main()
