from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load cleaned datasets.
    """

    return pd.read_csv(PROCESSED_DIR / filename)


def build_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    support: pd.DataFrame,
    marketing: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer-level analytical features.
    """

    customer_features = customers.copy()

    sales_summary = (
        transactions.groupby("customer_id")
        .agg(
            total_spend=(
                "sales_usd",
                "sum",
            ),
            total_orders=(
                "transaction_id",
                "count",
            ),
            average_order_value=(
                "sales_usd",
                "mean",
            ),
            total_profit=(
                "profit",
                "sum",
            ),
            last_purchase_date=(
                "order_date",
                "max",
            ),
        )
        .reset_index()
    )

    customer_features = customer_features.merge(
        sales_summary,
        on="customer_id",
        how="left",
    )

    support_summary = (
        support.groupby("customer_id")
        .agg(
            support_tickets=(
                "ticket_id",
                "count",
            ),
            average_satisfaction=(
                "customer_satisfaction",
                "mean",
            ),
        )
        .reset_index()
    )

    customer_features = customer_features.merge(
        support_summary,
        on="customer_id",
        how="left",
    )

    marketing_summary = (
        marketing.groupby("customer_id")
        .agg(
            total_campaigns=(
                "campaign_id",
                "count",
            ),
            total_clicks=(
                "clicks",
                "sum",
            ),
            average_engagement=(
                "engagement_score",
                "mean",
            ),
        )
        .reset_index()
    )

    customer_features = customer_features.merge(
        marketing_summary,
        on="customer_id",
        how="left",
    )

    numeric_columns = [
        "total_spend",
        "total_orders",
        "average_order_value",
        "total_profit",
        "support_tickets",
        "average_satisfaction",
        "total_campaigns",
        "total_clicks",
        "average_engagement",
    ]

    for column in numeric_columns:
        customer_features[column] = customer_features[column].fillna(0)

    customer_features["last_purchase_date"] = pd.to_datetime(
        customer_features["last_purchase_date"],
        errors="coerce",
    )

    customer_features["days_since_purchase"] = (
        pd.Timestamp.today() - customer_features["last_purchase_date"]
    ).dt.days

    customer_features["days_since_purchase"] = customer_features[
        "days_since_purchase"
    ].fillna(9999)

    return customer_features


def build_sales_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series sales features
    for forecasting models.
    """

    transactions = transactions.copy()

    transactions["order_date"] = pd.to_datetime(
        transactions["order_date"],
        errors="coerce",
    )

    transactions["month"] = transactions["order_date"].dt.to_period("M").astype(str)

    sales_features = (
        transactions.groupby(
            [
                "month",
                "country",
                "city",
            ]
        )
        .agg(
            total_sales=(
                "sales_usd",
                "sum",
            ),
            total_profit=(
                "profit",
                "sum",
            ),
            transactions_count=(
                "transaction_id",
                "count",
            ),
            average_order_value=(
                "sales_usd",
                "mean",
            ),
            units_sold=(
                "quantity",
                "sum",
            ),
        )
        .reset_index()
    )

    return sales_features


def build_product_features(
    products: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create product performance features.
    """

    sales_summary = (
        transactions.groupby("product_id")
        .agg(
            units_sold=(
                "quantity",
                "sum",
            ),
            total_revenue=(
                "sales_usd",
                "sum",
            ),
            total_profit=(
                "profit",
                "sum",
            ),
            transactions_count=(
                "transaction_id",
                "count",
            ),
            returned_orders=(
                "order_status",
                lambda x: (x == "Returned").sum(),
            ),
        )
        .reset_index()
    )

    product_features = products.merge(
        sales_summary,
        on="product_id",
        how="left",
    )

    product_features[
        [
            "units_sold",
            "total_revenue",
            "total_profit",
            "transactions_count",
            "returned_orders",
        ]
    ] = product_features[
        [
            "units_sold",
            "total_revenue",
            "total_profit",
            "transactions_count",
            "returned_orders",
        ]
    ].fillna(
        0
    )

    product_features["return_rate"] = product_features[
        "returned_orders"
    ] / product_features["transactions_count"].replace(0, 1)

    return product_features


def save_features(dataframe: pd.DataFrame, filename: str):
    """
    Save engineered features.
    """

    path = PROCESSED_DIR / filename

    dataframe.to_csv(
        path,
        index=False,
    )

    print(f"Saved {path}: {len(dataframe):,} rows")


def run_feature_engineering():
    """
    Execute complete feature engineering pipeline.
    """

    print("Loading cleaned datasets...")

    customers = load_processed_data("cleaned_customers.csv")

    transactions = load_processed_data("cleaned_transactions.csv")

    products = load_processed_data("cleaned_products.csv")

    support = load_processed_data("cleaned_support.csv")

    marketing = load_processed_data("cleaned_marketing.csv")

    print("Building customer features...")

    customer_features = build_customer_features(
        customers,
        transactions,
        support,
        marketing,
    )

    print("Building sales features...")

    sales_features = build_sales_features(transactions)

    print("Building product features...")

    product_features = build_product_features(
        products,
        transactions,
    )

    save_features(
        customer_features,
        "customer_features.csv",
    )

    save_features(
        sales_features,
        "sales_features.csv",
    )

    save_features(
        product_features,
        "product_features.csv",
    )

    print("Feature engineering completed successfully.")

    return {
        "customer_features": customer_features,
        "sales_features": sales_features,
        "product_features": product_features,
    }


if __name__ == "__main__":

    run_feature_engineering()


def save_features(dataframe: pd.DataFrame, filename: str):
    """
    Save engineered features.
    """

    path = PROCESSED_DIR / filename

    dataframe.to_csv(
        path,
        index=False,
    )

    print(f"Saved {path}: {len(dataframe):,} rows")


def run_feature_engineering():
    """
    Execute complete feature engineering pipeline.
    """

    print("Loading cleaned datasets...")

    customers = load_processed_data("cleaned_customers.csv")

    transactions = load_processed_data("cleaned_transactions.csv")

    products = load_processed_data("cleaned_products.csv")

    support = load_processed_data("cleaned_support.csv")

    marketing = load_processed_data("cleaned_marketing.csv")

    print("Building customer features...")

    customer_features = build_customer_features(
        customers,
        transactions,
        support,
        marketing,
    )

    print("Building sales features...")

    sales_features = build_sales_features(transactions)

    print("Building product features...")

    product_features = build_product_features(
        products,
        transactions,
    )

    save_features(
        customer_features,
        "customer_features.csv",
    )

    save_features(
        sales_features,
        "sales_features.csv",
    )

    save_features(
        product_features,
        "product_features.csv",
    )

    print("Feature engineering completed successfully.")

    return {
        "customer_features": customer_features,
        "sales_features": sales_features,
        "product_features": product_features,
    }


if __name__ == "__main__":

    run_feature_engineering()
