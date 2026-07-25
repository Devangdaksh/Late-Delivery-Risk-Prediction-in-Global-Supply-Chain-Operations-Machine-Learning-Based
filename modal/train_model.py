"""
APL Logistics — Late Delivery Risk Prediction
Training pipeline: feature engineering, model training, evaluation, artifact export.

IMPORTANT LEAKAGE NOTE
-----------------------
'Delivery Status' and 'Days for shipping (real)' are only known AFTER an order has
shipped/arrived. 'Delivery Status' in fact maps 1:1 onto the target
('Late delivery' <=> Late_delivery_risk=1), and 'Days for shipping (real)' minus
'Days for shipment (scheduled)' also perfectly separates the classes. Since the whole
point of this project is to flag risk BEFORE shipping, both columns are dropped from
the feature set. Using them would produce a model with ~100% accuracy that is
operationally useless (and dishonest).
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "APL_Logistics_Cleaned.csv"

LEAKY_COLS = ["Delivery Status", "Days for shipping (real)"]

# High-cardinality / identifier / redundant columns we don't feed to the model directly
DROP_COLS = [
    "Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode",
    "Customer Id", "Order Customer Id", "Customer City", "Order City",
    "Product Name", "Category Id", "Department Id",
    "Latitude", "Longitude", "Order State", "Customer State",
]

TARGET = "Late_delivery_risk"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Shipping pressure index: how many units must move per scheduled day
    df["shipping_pressure_index"] = (
        df["Order Item Quantity"] / (df["Days for shipment (scheduled)"] + 1)
    )

    # Mode risk flag: express-type modes vs standard
    express_modes = {"Same Day", "First Class"}
    df["express_mode_flag"] = df["Shipping Mode"].isin(express_modes).astype(int)

    # Order complexity score: quantity scaled by discount depth (deeper discounts +
    # larger quantities historically correlate with more handling complexity)
    df["order_complexity_score"] = (
        df["Order Item Quantity"] * (1 + df["Order Item Discount Rate"])
    )

    # Scheduled speed tier (very tight vs relaxed schedules)
    df["tight_schedule_flag"] = (df["Days for shipment (scheduled)"] <= 1).astype(int)

    # Discount depth relative to price
    df["discount_to_price_ratio"] = (
        df["Order Item Discount"] / df["Order Item Product Price"].replace(0, np.nan)
    ).fillna(0)

    return df


def add_regional_congestion(df_train, df_other_list, col="Order Region"):
    """Target-encode historical late-rate per region, fit strictly on train split."""
    rate_map = df_train.groupby(col)[TARGET].mean()
    global_rate = df_train[TARGET].mean()
    out = []
    for d in [df_train] + df_other_list:
        d = d.copy()
        d["regional_congestion_index"] = d[col].map(rate_map).fillna(global_rate)
        out.append(d)
    return out, rate_map, global_rate


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df = engineer_features(df)

    # Train / val / test split (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET], random_state=RANDOM_STATE
    )
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    (train_df, val_df, test_df), region_rate_map, global_rate = add_regional_congestion(
        train_df, [val_df, test_df], col="Order Region"
    )

    categorical_cols = [
        "Type", "Category Name", "Customer Country", "Customer Segment",
        "Department Name", "Market", "Order Country", "Order Region",
        "Order Status", "Shipping Mode",
    ]
    numeric_cols = [
        "Days for shipment (scheduled)", "Benefit per order", "Sales per customer",
        "Order Item Discount", "Order Item Discount Rate", "Order Item Product Price",
        "Order Item Profit Ratio", "Order Item Quantity", "Sales", "Order Item Total",
        "Order Profit Per Order", "Product Price",
        "shipping_pressure_index", "express_mode_flag", "order_complexity_score",
        "tight_schedule_flag", "discount_to_price_ratio", "regional_congestion_index",
    ]

    feature_cols = categorical_cols + numeric_cols

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_val, y_val = val_df[feature_cols], val_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=14, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            )),
        ]),
    }

    results = {}
    fitted = {}
    for name, pipe in models.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": roc_auc_score(y_val, val_proba),
            "precision": precision_score(y_val, val_pred),
            "recall": recall_score(y_val, val_pred),
            "f1": f1_score(y_val, val_pred),
        }
        fitted[name] = pipe
        print(name, results[name])

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Final test-set evaluation of the best model
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": roc_auc_score(y_test, test_proba),
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    print("Test metrics:", test_metrics)

    # Feature importance (best model, if tree-based; else coefficients)
    ohe = best_model.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Save everything the Streamlit app needs
    joblib.dump(best_model, "late_delivery_model.joblib")
    joblib.dump(region_rate_map, "region_rate_map.joblib")

    artifact = {
        "best_model_name": best_name,
        "global_late_rate": float(global_rate),
        "model_comparison": results,
        "test_metrics": test_metrics,
        "feature_importance": fi_df.head(20).to_dict(orient="records"),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
    }
    with open("model_artifacts.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print("\nSaved: late_delivery_model.joblib, region_rate_map.joblib, model_artifacts.json")
    print("\nTop 10 feature importances:")
    print(fi_df.head(10))


if __name__ == "__main__":
    main()
