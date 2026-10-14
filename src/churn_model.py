from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def load_customer_features() -> pd.DataFrame:
    """
    Load engineered customer features.
    """

    return pd.read_csv(PROCESSED_DIR / "customer_features.csv")


def create_churn_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn labels based on customer behaviour.

    1 = likely churn
    0 = active customer
    """

    dataframe = dataframe.copy()

    dataframe["churn"] = (
        (dataframe["days_since_purchase"] > 180)
        | (dataframe["total_orders"] <= 1)
        | (dataframe["average_satisfaction"] < 3)
    ).astype(int)

    return dataframe


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def prepare_training_data(dataframe: pd.DataFrame):
    """
    Prepare features and target variable.
    """

    feature_columns = [
        "age",
        "loyalty_score",
        "total_spend",
        "total_orders",
        "average_order_value",
        "total_profit",
        "days_since_purchase",
        "support_tickets",
        "average_satisfaction",
        "total_campaigns",
        "total_clicks",
        "average_engagement",
    ]

    X = dataframe[feature_columns]

    y = dataframe["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    y_probability,
):

    return {
        "model": model_name,
        "accuracy": round(
            accuracy_score(
                y_true,
                y_pred,
            ),
            4,
        ),
        "precision": round(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "f1_score": round(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(
                y_true,
                y_probability,
            ),
            4,
        ),
    }


def train_churn_models():
    """
    Train churn prediction models.
    """

    print("Loading customer features...")

    data = load_customer_features()

    print("Creating churn target...")

    data = create_churn_target(data)

    X_train, X_test, y_train, y_test = prepare_training_data(data)

    results = []

    logistic_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    print("Training Logistic Regression...")

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_predictions = logistic_model.predict(X_test)

    logistic_probability = logistic_model.predict_proba(X_test)[:, 1]

    results.append(
        evaluate_predictions(
            "Logistic Regression",
            y_test,
            logistic_predictions,
            logistic_probability,
        )
    )

    print("Training Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    random_forest.fit(
        X_train,
        y_train,
    )

    rf_predictions = random_forest.predict(X_test)

    rf_probability = random_forest.predict_proba(X_test)[:, 1]

    results.append(
        evaluate_predictions(
            "Random Forest",
            y_test,
            rf_predictions,
            rf_probability,
        )
    )

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        PROCESSED_DIR / "churn_model_metrics.csv",
        index=False,
    )

    print(metrics)

    return (
        logistic_model,
        random_forest,
        metrics,
    )


def generate_churn_predictions():
    """
    Generate churn predictions for all customers.
    """

    data = load_customer_features()

    data = create_churn_target(data)

    (
        logistic_model,
        random_forest,
        metrics,
    ) = train_churn_models()

    feature_columns = [
        "age",
        "loyalty_score",
        "total_spend",
        "total_orders",
        "average_order_value",
        "total_profit",
        "days_since_purchase",
        "support_tickets",
        "average_satisfaction",
        "total_campaigns",
        "total_clicks",
        "average_engagement",
    ]

    X = data[feature_columns]

    probabilities = random_forest.predict_proba(X)[:, 1]

    data["churn_probability"] = probabilities.round(4)

    data["risk_level"] = data["churn_probability"].apply(
        lambda x: (
            "High Risk" if x >= 0.7 else ("Medium Risk" if x >= 0.4 else "Low Risk")
        )
    )

    output_columns = [
        "customer_id",
        "customer_name",
        "country",
        "customer_segment",
        "total_spend",
        "total_orders",
        "days_since_purchase",
        "average_satisfaction",
        "churn_probability",
        "risk_level",
    ]

    predictions = data[output_columns]

    predictions.to_csv(
        PROCESSED_DIR / "customer_churn_predictions.csv",
        index=False,
    )

    print("Saved customer_churn_predictions.csv")

    return predictions


if __name__ == "__main__":

    generate_churn_predictions()
