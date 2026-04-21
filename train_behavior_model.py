"""
train_behavior_model.py
=======================
Instructions:
    1. Install dependencies:
           pip install pandas scikit-learn joblib numpy
    2. Run:
           python train_behavior_model.py
    3. Output: behavior_model.pkl  +  behavior_scaler.pkl
Notes:
    - Uses RandomForestClassifier (scikit-learn only -- no xgboost needed).
    - Synthetic data is generated for demo purposes.
    - Replace `generate_synthetic_data()` with your real dataset loading logic.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -- Configuration ------------------------------------------------------------
MODEL_OUTPUT   = "behavior_model.pkl"
SCALER_OUTPUT  = "behavior_scaler.pkl"
RANDOM_STATE   = 42
N_SAMPLES      = 5000

# -- Synthetic Data Generator (replace with real data) ------------------------
def generate_synthetic_data(n=N_SAMPLES, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    data = {
        "account_age_days"       : rng.integers(1, 1825, n),
        "total_orders"           : rng.integers(1, 500, n),
        "total_refund_requests"  : rng.integers(0, 50, n),
        "order_value"            : rng.uniform(50, 2000, n).round(2),
        "num_items"              : rng.integers(1, 10, n),
        "is_peak_hour"           : rng.integers(0, 2, n),
        "delivery_time_min"      : rng.integers(15, 120, n),
        "distance_km"            : rng.uniform(0.5, 30, n).round(2),
        "payment_method"         : rng.choice(["card", "upi", "cod", "wallet"], n),
        "is_new_address"         : rng.integers(0, 2, n),
        "promo_applied"          : rng.integers(0, 2, n),
        "restaurant_rating"      : rng.uniform(1, 5, n).round(1),
        "driver_rating"          : rng.uniform(1, 5, n).round(1),
        "prev_complaints_30d"    : rng.integers(0, 10, n),
        "prev_fraud_flags"       : rng.integers(0, 5, n),
        "refund_amount_30d"      : rng.uniform(0, 5000, n).round(2),
        "days_since_last_refund" : rng.integers(0, 365, n),
    }
    df = pd.DataFrame(data)
    fraud_score = (
        (df["total_refund_requests"] / (df["total_orders"] + 1) > 0.15).astype(int)
        + (df["prev_fraud_flags"] > 0).astype(int)
        + (df["prev_complaints_30d"] >= 3).astype(int)
        + (df["account_age_days"] < 30).astype(int)
        + (df["is_new_address"] == 1).astype(int)
        + (df["promo_applied"] == 1).astype(int)
    )
    df["label"] = (fraud_score >= 3).astype(int)
    return df

# -- Feature Engineering ------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complaint_ratio"]      = df["total_refund_requests"] / (df["total_orders"] + 1)
    df["refund_per_complaint"] = df["refund_amount_30d"] / (df["prev_complaints_30d"] + 1)
    df["new_addr_promo"]       = df["is_new_address"] * df["promo_applied"]
    df["value_per_item"]       = df["order_value"] / (df["num_items"] + 1)
    df["recent_refund_flag"]   = (df["days_since_last_refund"] < 7).astype(int)
    df["is_new_account"]       = (df["account_age_days"] < 60).astype(int)
    return df

# -- Preprocessing ------------------------------------------------------------
def preprocess(df: pd.DataFrame, scaler=None, fit_scaler=True):
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=["payment_method"], drop_first=False)
    for col in ["payment_method_card", "payment_method_cod",
                "payment_method_upi", "payment_method_wallet"]:
        if col not in df.columns:
            df[col] = 0
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].fillna(0)
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler, feature_cols

# -- Main Training ------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic training data ...")
    df = generate_synthetic_data()
    print(f"Dataset shape: {df.shape}  |  Fraud rate: {df['label'].mean():.2%}")

    y = df["label"].values
    X_scaled, scaler, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # class_weight='balanced' handles fraud/genuine imbalance automatically
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining RandomForest model ...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"\nBehavior model saved to : {MODEL_OUTPUT}")
    print(f"Scaler saved to         : {SCALER_OUTPUT}")