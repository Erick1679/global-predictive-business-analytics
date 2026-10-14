from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")


def load_customer_features() -> pd.DataFrame:
    """
    Load engineered customer features.
    """

    return pd.read_csv(PROCESSED_DIR / "customer_features.csv")


def calculate_clv(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate estimated customer lifetime value.

    CLV is based on:
    - Historical spend
    - Purchase frequency
    - Profit contribution
    - Loyalty
    - Engagement
    """

    dataframe = dataframe.copy()

    dataframe["customer_clv"] = (
        (dataframe["total_spend"] * 0.45)
        + (dataframe["total_profit"] * 0.35)
        + (dataframe["total_orders"] * dataframe["average_order_value"] * 0.15)
        + (dataframe["loyalty_score"] * 50 * 0.05)
    )

    dataframe["customer_clv"] = dataframe["customer_clv"].round(2)

    return dataframe


def assign_clv_segment(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize customers by lifetime value.
    """

    dataframe = dataframe.copy()

    q1 = dataframe["customer_clv"].quantile(0.25)

    q2 = dataframe["customer_clv"].quantile(0.50)

    q3 = dataframe["customer_clv"].quantile(0.75)

    def segment(value):

        if value >= q3:
            return "Platinum"

        elif value >= q2:
            return "Gold"

        elif value >= q1:
            return "Silver"

        else:
            return "Bronze"

    dataframe["clv_segment"] = dataframe["customer_clv"].apply(segment)

    return dataframe


def create_clv_predictions():
    """
    Generate customer lifetime value predictions.
    """

    print("Loading customer features...")

    customers = load_customer_features()

    print("Calculating customer lifetime value...")

    customers = calculate_clv(customers)

    print("Assigning CLV segments...")

    customers = assign_clv_segment(customers)

    customers = customers.sort_values(
        by="customer_clv",
        ascending=False,
    )

    output_columns = [
        "customer_id",
        "customer_name",
        "country",
        "customer_segment",
        "total_spend",
        "total_orders",
        "total_profit",
        "customer_clv",
        "clv_segment",
    ]

    predictions = customers[output_columns]

    predictions.to_csv(
        PROCESSED_DIR / "customer_clv_predictions.csv",
        index=False,
    )

    summary = (
        customers.groupby("clv_segment")
        .agg(
            customers=(
                "customer_id",
                "count",
            ),
            average_clv=(
                "customer_clv",
                "mean",
            ),
            total_value=(
                "customer_clv",
                "sum",
            ),
            average_spend=(
                "total_spend",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        [
            "average_clv",
            "total_value",
            "average_spend",
        ]
    ] = summary[
        [
            "average_clv",
            "total_value",
            "average_spend",
        ]
    ].round(2)

    summary.to_csv(
        PROCESSED_DIR / "clv_segment_summary.csv",
        index=False,
    )

    print("Saved customer_clv_predictions.csv")

    print("Saved clv_segment_summary.csv")

    print(summary)

    return (
        predictions,
        summary,
    )


if __name__ == "__main__":

    create_clv_predictions()
