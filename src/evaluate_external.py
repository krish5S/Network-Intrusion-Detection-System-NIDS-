"""Evaluate the saved CICIDS2017 XGBoost model on an external CICFlowMeter CSV."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/external/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
DATASET_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
)

# CSE-CIC-IDS2018 uses CICFlowMeter V3 header abbreviations. The values retain
# the same feature definitions; identifiers (Protocol and Timestamp) are not
# model inputs.
SOURCE_NAMES = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/cse_cic_ids2018_external_evaluation.json",
    )
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if not args.data.is_file():
        raise SystemExit(f"Dataset CSV not found: {args.data}")

    model_path = ROOT / "models/xgb_model.joblib"
    scaler_path = ROOT / "models/scaler.joblib"
    metadata_path = ROOT / "models/metadata.json"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(metadata_path.read_text())
    features = metadata["features"]

    if len(features) != 77 or model.n_features_in_ != len(features):
        raise SystemExit("Saved metadata and XGBoost feature count do not match")

    df = pd.read_csv(args.data, low_memory=False)
    df.columns = df.columns.str.strip()
    raw_rows = len(df)
    if "Label" not in df.columns:
        raise SystemExit("External CSV is missing the Label column")
    raw_label_counts = {
        str(label): int(count)
        for label, count in df["Label"].astype("string").fillna("<missing>").value_counts().items()
    }

    original_source_names = {feature: SOURCE_NAMES.get(feature, feature) for feature in features}
    missing = [source for source in original_source_names.values() if source not in df.columns]
    if missing:
        raise SystemExit(
            "Cannot evaluate without fabricating model inputs. Missing CICFlowMeter "
            f"features: {missing}"
        )

    label_text = df["Label"].astype("string").str.strip()
    # Some source files repeat their header in the body; discard those rows
    # instead of treating the literal word "Label" as an attack category.
    label_valid = label_text.notna() & label_text.ne("") & label_text.str.casefold().ne("label")
    y_all = label_text.str.casefold().ne("benign").astype("int8")

    # Keep one record per feature vector and label, matching the project's
    # original exact-duplicate removal without allowing IDs/timestamps to make
    # otherwise identical feature records appear unique.
    mapped_features = df[[original_source_names[f] for f in features]].copy()
    mapped_features.columns = features
    for feature in features:
        mapped_features[feature] = pd.to_numeric(mapped_features[feature], errors="coerce")
    mapped_features = mapped_features.replace([np.inf, -np.inf], np.nan)

    valid_rows = label_valid & mapped_features.notna().all(axis=1)
    invalid_rows = int((~valid_rows).sum())
    X = mapped_features.loc[valid_rows].reset_index(drop=True)
    y = y_all.loc[valid_rows].reset_index(drop=True)
    valid_class_counts = {
        "benign": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
    }

    before_dedupe = len(X)
    dedupe_frame = X.copy()
    dedupe_frame["__binary_label__"] = y
    keep = ~dedupe_frame.duplicated(keep="first")
    duplicate_rows = int((~keep).sum())
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    if y.nunique() != 2:
        raise SystemExit(f"Expected benign and attack flows; found classes {y.value_counts().to_dict()}")

    classes = np.asarray(model.classes_)
    attack_indexes = np.flatnonzero(classes == 1)
    benign_indexes = np.flatnonzero(classes == 0)
    if len(attack_indexes) != 1 or len(benign_indexes) != 1:
        raise SystemExit(f"Expected model classes [0, 1], got {classes.tolist()}")

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)
    p_attack = probabilities[:, attack_indexes[0]]
    y_pred = (p_attack >= args.threshold).astype("int8")

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    threshold_sensitivity = []
    for candidate in (0.05, 0.10, 0.25, 0.50, 0.75, 0.95):
        candidate_pred = (p_attack >= candidate).astype("int8")
        c_tn, c_fp, c_fn, c_tp = confusion_matrix(y, candidate_pred, labels=[0, 1]).ravel()
        threshold_sensitivity.append({
            "threshold": candidate,
            "attack_precision": float(precision_score(y, candidate_pred, zero_division=0)),
            "attack_recall": float(recall_score(y, candidate_pred, zero_division=0)),
            "false_positive_rate": float(c_fp / (c_fp + c_tn)) if c_fp + c_tn else None,
            "tn": int(c_tn), "fp": int(c_fp), "fn": int(c_fn), "tp": int(c_tp),
        })
    label_counts = {
        "benign": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
    }
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=["Benign", "Attack"],
        output_dict=True, zero_division=0,
    )
    model_params = model.get_params()
    result = {
        "evaluation_date": date.today().isoformat(),
        "dataset": "CSE-CIC-IDS2018",
        "dataset_source": DATASET_URL,
        "dataset_file": args.data.name,
        "dataset_sha256": sha256_file(args.data),
        "raw_label_counts": raw_label_counts,
        "preprocessing": {
            "raw_rows": raw_rows,
            "invalid_rows_dropped": invalid_rows,
            "valid_rows_before_deduplication": before_dedupe,
            "exact_duplicate_rows_dropped": duplicate_rows,
            "rows_scored": len(X),
            "label_mapping": {"Benign": 0, "other_nonempty_nonheader_labels": 1},
            "feature_count": len(features),
            "feature_mapping": original_source_names,
            "scaler_refit": False,
            "model_refit": False,
        },
        "model": {
            "artifact": str(model_path.relative_to(ROOT)),
            "scaler": str(scaler_path.relative_to(ROOT)),
            "xgboost_version": __import__("xgboost").__version__,
            "objective": model_params.get("objective"),
            "n_estimators": model_params.get("n_estimators"),
            "max_depth": model_params.get("max_depth"),
            "learning_rate": model_params.get("learning_rate"),
            "classes": classes.tolist(),
        },
        "decision_threshold": args.threshold,
        "class_counts_after_cleaning_before_deduplication": valid_class_counts,
        "class_counts_scored": label_counts,
        "confusion_matrix_labels_actual_rows_predicted_columns": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "metrics": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_attack": float(precision_score(y, y_pred, zero_division=0)),
            "recall_attack": float(recall_score(y, y_pred, zero_division=0)),
            "f1_attack": float(f1_score(y, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "roc_auc": float(roc_auc_score(y, p_attack)),
            "pr_auc_average_precision": float(average_precision_score(y, p_attack)),
            "brier_score_attack_probability": float(np.mean((p_attack - y.to_numpy()) ** 2)),
        },
        "threshold_sensitivity": threshold_sensitivity,
        "per_class_report": report,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
