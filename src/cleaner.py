from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the raw data directory.
    """

    path = RAW_DIR / filename

    return pd.read_csv(path)


def generate_quality_report(dataframe: pd.DataFrame, dataset_name: str) -> dict:
    """
    Generate a data quality summary for a dataset.
    """

    return {
        "dataset": dataset_name,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def clean_dates(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert date columns into proper datetime format.
    """

    dataframe = dataframe.copy()

    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records.
    """

    return dataframe.drop_duplicates()


def clean_transactions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales transaction data.
    """

    dataframe = dataframe.copy()

    dataframe = clean_dates(
        dataframe,
        ["order_date"],
    )

    dataframe = remove_duplicates(dataframe)

    dataframe = dataframe[dataframe["quantity"] > 0]

    dataframe = dataframe[dataframe["net_sales"] >= 0]

    dataframe["profit_margin"] = dataframe["profit_margin"].fillna(0)

    return dataframe


def clean_customers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean customer master data.
    """

    dataframe = dataframe.copy()

    dataframe = remove_duplicates(dataframe)

    text_columns = [
        "country",
        "city",
        "customer_segment",
        "preferred_channel",
        "preferred_category",
    ]

    for column in text_columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype(str).str.strip()

    return dataframe


def clean_products(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product catalogue data.
    """

    dataframe = dataframe.copy()

    dataframe = remove_duplicates(dataframe)

    dataframe = dataframe[dataframe["selling_price"] > 0]

    dataframe = dataframe[dataframe["cost_price"] >= 0]

    return dataframe


def clean_marketing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean marketing campaign data.
    """

    dataframe = dataframe.copy()

    dataframe = clean_dates(
        dataframe,
        ["campaign_date"],
    )

    dataframe = remove_duplicates(dataframe)

    dataframe["clicks"] = dataframe["clicks"].clip(lower=0)

    dataframe["impressions"] = dataframe["impressions"].clip(lower=0)

    return dataframe


def clean_support(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean customer support data.
    """

    dataframe = dataframe.copy()

    dataframe = clean_dates(
        dataframe,
        ["created_date"],
    )

    dataframe = remove_duplicates(dataframe)

    dataframe["customer_satisfaction"] = dataframe["customer_satisfaction"].clip(1, 5)

    return dataframe


def clean_inventory(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inventory records.
    """

    dataframe = dataframe.copy()

    dataframe = clean_dates(
        dataframe,
        ["inventory_date"],
    )

    dataframe = remove_duplicates(dataframe)

    dataframe["stock_on_hand"] = dataframe["stock_on_hand"].clip(lower=0)

    return dataframe


def run_cleaning_pipeline() -> Dict[str, pd.DataFrame]:
    """
    Execute the complete data cleaning workflow.
    """

    datasets = {}

    quality_records = []

    files = {
        "customers": "customers.csv",
        "products": "products.csv",
        "transactions": "transactions.csv",
        "marketing": "marketing.csv",
        "support": "support.csv",
        "inventory": "inventory.csv",
    }

    for name, filename in files.items():

        print(f"Loading {filename}")

        datasets[name] = load_dataset(filename)

    datasets["customers"] = clean_customers(datasets["customers"])

    datasets["products"] = clean_products(datasets["products"])

    datasets["transactions"] = clean_transactions(datasets["transactions"])

    datasets["marketing"] = clean_marketing(datasets["marketing"])

    datasets["support"] = clean_support(datasets["support"])

    datasets["inventory"] = clean_inventory(datasets["inventory"])

    for name, dataframe in datasets.items():

        quality_records.append(
            generate_quality_report(
                dataframe,
                name,
            )
        )

        output_file = PROCESSED_DIR / f"cleaned_{name}.csv"

        dataframe.to_csv(
            output_file,
            index=False,
        )

        print(f"Saved {output_file}")

    quality_report = pd.DataFrame(quality_records)

    quality_report.to_csv(
        PROCESSED_DIR / "data_quality_report.csv",
        index=False,
    )

    print("Saved data_quality_report.csv")

    return datasets


if __name__ == "__main__":

    run_cleaning_pipeline()
