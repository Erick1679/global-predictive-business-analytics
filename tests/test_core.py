import pandas as pd

from src.cleaner import clean_dataframe
from src.data_generator import generate_sample_data
from src.feature_engineering import build_features
from src.churn_model import train_churn_model
from src.clv_model import train_clv_model
from src.forecasting import forecast_sales
from src.pipeline import run_pipeline


def test_generate_sample_data_creates_expected_tables():
    data = generate_sample_data()
    assert set(data) == {"customers", "products", "transactions", "marketing", "support", "inventory"}
    assert all(isinstance(frame, pd.DataFrame) for frame in data.values())


def test_clean_and_feature_building_produce_valid_customer_features():
    data = generate_sample_data()
    cleaned = clean_dataframe(data["transactions"], date_columns=["transaction_date"])
    features = build_features(cleaned)

    assert {"customer_id", "total_revenue", "avg_order_value", "total_orders"}.issubset(features.columns)
    assert features["customer_id"].nunique() == len(features)
    assert cleaned["transaction_date"].dtype.kind == "M"


def test_model_training_and_pipeline_smoke():
    data = generate_sample_data()
    cleaned = clean_dataframe(data["transactions"], date_columns=["transaction_date"])
    features = build_features(cleaned)
    features["is_churned"] = (features["total_revenue"] <= features["total_revenue"].median()).astype(int)
    features["customer_lifetime_value"] = features["total_revenue"] * 1.25

    churn_model, churn_metrics = train_churn_model(features, target_column="is_churned")
    clv_model, clv_metrics = train_clv_model(features, target_column="customer_lifetime_value")

    assert set(churn_metrics).issuperset({"accuracy", "roc_auc"})
    assert set(clv_metrics).issuperset({"mae", "rmse"})
    assert callable(getattr(churn_model, "predict", None))
    assert callable(getattr(clv_model, "predict", None))

    forecast = forecast_sales(cleaned[["transaction_date", "revenue"]], periods=3)
    assert len(forecast) == 3

    result = run_pipeline(generate_data=True, output_dir="data/raw")
    assert "forecast" in result
    assert len(result["forecast"]) == 6
