from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")


def load_sales_features() -> pd.DataFrame:
    """
    Load engineered sales forecasting data.
    """

    dataframe = pd.read_csv(PROCESSED_DIR / "sales_features.csv")

    dataframe["month"] = pd.to_datetime(
        dataframe["month"],
        errors="coerce",
    )

    return dataframe


def prepare_forecasting_data(dataframe: pd.DataFrame):
    """
    Prepare time-series regression features.
    """

    dataframe = dataframe.copy()

    dataframe = dataframe.sort_values("month")

    dataframe["month_number"] = dataframe["month"].dt.month

    dataframe["year"] = dataframe["month"].dt.year

    dataframe["quarter"] = dataframe["month"].dt.quarter

    dataframe["lag_1_month_sales"] = dataframe["total_sales"].shift(1)

    dataframe["lag_3_month_sales"] = dataframe["total_sales"].shift(3)

    dataframe["rolling_3_month_sales"] = dataframe["total_sales"].rolling(3).mean()

    dataframe = dataframe.dropna()

    features = [
        "month_number",
        "year",
        "quarter",
        "transactions_count",
        "average_order_value",
        "units_sold",
        "lag_1_month_sales",
        "lag_3_month_sales",
        "rolling_3_month_sales",
    ]

    X = dataframe[features]

    y = dataframe["total_sales"]

    return (
        dataframe,
        X,
        y,
    )


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def train_forecasting_model():
    """
    Train sales forecasting model.
    """

    print("Loading sales features...")

    sales_data = load_sales_features()

    print("Preparing forecasting data...")

    prepared_data, X, y = prepare_forecasting_data(sales_data)

    split_index = int(len(prepared_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Training Random Forest forecasting model...")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

    metrics = pd.DataFrame(
        [
            {
                "model": "Random Forest Regressor",
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE": round(mape, 2),
            }
        ]
    )

    metrics.to_csv(
        PROCESSED_DIR / "forecast_metrics.csv",
        index=False,
    )

    forecast_results = prepared_data.iloc[split_index:].copy()

    forecast_results["actual_sales"] = y_test.values

    forecast_results["predicted_sales"] = predictions

    forecast_results.to_csv(
        PROCESSED_DIR / "sales_forecast_predictions.csv",
        index=False,
    )

    print(metrics)

    print("Forecasting model completed.")

    return (
        model,
        metrics,
        forecast_results,
    )


def forecast_future_sales(
    model,
    months_ahead: int = 6,
):
    """
    Generate future sales forecasts.

    Uses the latest known sales behaviour
    to estimate upcoming months.
    """

    sales_data = load_sales_features()

    prepared_data, X, y = prepare_forecasting_data(sales_data)

    latest_row = prepared_data.iloc[-1].copy()

    future_predictions = []

    last_date = latest_row["month"]

    recent_sales = list(prepared_data["total_sales"].tail(3))

    for i in range(1, months_ahead + 1):

        future_date = last_date + pd.DateOffset(months=i)

        future_input = pd.DataFrame(
            [
                {
                    "month_number": future_date.month,
                    "year": future_date.year,
                    "quarter": future_date.quarter,
                    "transactions_count": prepared_data["transactions_count"].mean(),
                    "average_order_value": prepared_data["average_order_value"].mean(),
                    "units_sold": prepared_data["units_sold"].mean(),
                    "lag_1_month_sales": recent_sales[-1],
                    "lag_3_month_sales": np.mean(recent_sales),
                    "rolling_3_month_sales": np.mean(recent_sales),
                }
            ]
        )

        prediction = model.predict(future_input)[0]

        future_predictions.append(
            {
                "forecast_month": future_date,
                "predicted_sales": round(
                    prediction,
                    2,
                ),
            }
        )

        recent_sales.append(prediction)

        recent_sales = recent_sales[-3:]

    forecast = pd.DataFrame(future_predictions)

    forecast.to_csv(
        PROCESSED_DIR / "future_sales_forecast.csv",
        index=False,
    )

    print("Saved future_sales_forecast.csv")

    return forecast


if __name__ == "__main__":

    model, metrics, results = train_forecasting_model()

    forecast_future_sales(model)
