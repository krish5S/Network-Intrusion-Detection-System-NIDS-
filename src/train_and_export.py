"""
Export trained NIDS models, scaler, feature metadata, and a test traffic sample.
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def prepare_data(data_dir="data", sample_benign_per_file=20000):
    files = sorted(Path(data_dir).glob("*.csv"))
    print(f"Found {len(files)} CSV files in {data_dir}...")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        if "Fwd Header Length.1" in df.columns:
            df = df.drop(columns=["Fwd Header Length.1"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
        df["Label"] = df["Label"].astype(str).str.replace(r"[^\x00-\x7F]+", "-", regex=True)

        attacks = df[df["Label"] != "BENIGN"]
        benign = df[df["Label"] == "BENIGN"]
        if len(benign) > sample_benign_per_file:
            benign = benign.sample(n=sample_benign_per_file, random_state=42)
        dfs.append(pd.concat([attacks, benign]))

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return combined

def main():
    start_time = time.time()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("Loading and preparing dataset...")
    df = prepare_data()
    print(f"Dataset shape: {df.shape}")

    features = [c for c in df.columns if c != "Label"]
    X = df[features]
    labels = df["Label"]
    y_binary = (labels != "BENIGN").astype(int)

    print(f"Splitting 80/20 stratified on labels...")
    X_train, X_test, y_train, y_test, lab_train, lab_test = train_test_split(
        X, y_binary, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train high-performance XGBoost
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    print("\n--- XGBoost Test Evaluation ---")
    print(confusion_matrix(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb, digits=4))

    # Train Random Forest (50 trees for fast loading and low memory in Streamlit)
    print("Training Random Forest Classifier (50 trees)...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=16, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    print("\n--- Random Forest Test Evaluation ---")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, digits=4))

    # Save artifacts
    print("Saving models and scaler to models/...")
    joblib.dump(xgb_model, models_dir / "xgb_model.joblib", compress=3)
    joblib.dump(rf_model, models_dir / "rf_model.joblib", compress=3)
    joblib.dump(scaler, models_dir / "scaler.joblib", compress=3)

    # Save feature names and metadata
    feature_meta = {
        "features": features,
        "n_features": len(features),
        "attack_types": sorted(list(labels.unique()))
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # Create a realistic test sample CSV for the UI (1,500 balanced flows with both benign and various attacks)
    print("Generating demo traffic samples for the frontend...")
    demo_sample_rows = []
    # Get a balanced mix of each attack type + benign from test set
    test_df = X_test.copy()
    test_df["Label"] = lab_test.values

    for attack in test_df["Label"].unique():
        sub = test_df[test_df["Label"] == attack]
        n_take = min(len(sub), 120 if attack == "BENIGN" else (80 if len(sub) > 80 else len(sub)))
        demo_sample_rows.append(sub.sample(n=n_take, random_state=42))

    demo_df = pd.concat(demo_sample_rows, ignore_index=True).sample(frac=1.0, random_state=42)
    sample_path = Path("data") / "sample_test_traffic.csv"
    demo_df.to_csv(sample_path, index=False)
    print(f"Saved {len(demo_df)} sample traffic flows to {sample_path}")

    print(f"\nDone in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
