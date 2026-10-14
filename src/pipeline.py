from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .cleaner import clean_dataframe
from .clv_model import train_clv_model
from .data_generator import save_sample_data
from .feature_engineering import build_features
from .forecasting import forecast_sales
from .churn_model import train_churn_model


def run_pipeline(
    *,
    output_dir: str | Path = "data/raw",
    generate_data: bool = True,
    customer_feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Run the end-to-end predictive analytics workflow."""
    if generate_data:
        save_sample_data(output_dir)

    customer_data = pd.read_csv(Path(output_dir) / "customers.csv")
    transactions = pd.read_csv(Path(output_dir) / "transactions.csv")
    products = pd.read_csv(Path(output_dir) / "products.csv")

    transactions = clean_dataframe(transactions, date_columns=["transaction_date"])
    joined = transactions.merge(customer_data[["customer_id", "segment", "region", "channel", "tenure_months"]], on="customer_id", how="left")
    joined = joined.merge(products[["product_id", "category", "price"]], on="product_id", how="left")
    joined["is_churned"] = (joined["customer_id"].astype(str).str.contains("CUST")).astype(int)
    joined["customer_lifetime_value"] = joined["revenue"] * 1.25

    features = build_features(joined)
    features["is_churned"] = (features["total_revenue"] <= features["total_revenue"].median()).astype(int)
    features["customer_lifetime_value"] = features["total_revenue"] * 1.25

    if customer_feature_columns is not None:
        features = features[customer_feature_columns]
    else:
        features = features.drop(columns=["first_purchase", "last_purchase"], errors="ignore")

    churn_model, churn_metrics = train_churn_model(features, target_column="is_churned")
    clv_model, clv_metrics = train_clv_model(features, target_column="customer_lifetime_value")
    forecast = forecast_sales(joined[["transaction_date", "revenue"]], periods=6)

    return {
        "customer_data": customer_data,
        "transactions": transactions,
        "features": features,
        "churn_model": churn_model,
        "churn_metrics": churn_metrics,
        "clv_model": clv_model,
        "clv_metrics": clv_metrics,
        "forecast": forecast,
    }


build_pipeline = run_pipeline
