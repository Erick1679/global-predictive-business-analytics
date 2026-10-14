from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)


OUTPUT_DIR = Path("data/raw")


START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2026-12-31")


COUNTRIES = {
    "United States": [
        "New York",
        "California",
        "Texas",
        "Florida",
        "Illinois",
        "Washington",
    ],
    "United Kingdom": [
        "London",
        "Manchester",
        "Birmingham",
    ],
    "Germany": [
        "Berlin",
        "Munich",
        "Frankfurt",
    ],
    "France": [
        "Paris",
        "Lyon",
        "Marseille",
    ],
    "Netherlands": [
        "Amsterdam",
        "Rotterdam",
    ],
    "Spain": [
        "Madrid",
        "Barcelona",
    ],
    "Italy": [
        "Rome",
        "Milan",
    ],
    "Sweden": [
        "Stockholm",
        "Gothenburg",
    ],
}


CURRENCIES = {
    "United States": ("USD", 1.0),
    "United Kingdom": ("GBP", 0.79),
    "Germany": ("EUR", 0.92),
    "France": ("EUR", 0.92),
    "Netherlands": ("EUR", 0.92),
    "Spain": ("EUR", 0.92),
    "Italy": ("EUR", 0.92),
    "Sweden": ("SEK", 10.5),
}


PRODUCT_CATEGORIES = {
    "Electronics": [
        "Smartphones",
        "Laptops",
        "Accessories",
        "Gaming",
    ],
    "Fashion": [
        "Clothing",
        "Footwear",
        "Luxury",
    ],
    "Home": [
        "Furniture",
        "Kitchen",
        "Appliances",
    ],
    "Sports": [
        "Fitness",
        "Outdoor",
        "Equipment",
    ],
    "Beauty": [
        "Skincare",
        "Cosmetics",
        "Personal Care",
    ],
}


def random_date():
    return fake.date_between(
        start_date=START_DATE.date(),
        end_date=END_DATE.date(),
    )


def random_country():
    return fake.random_element(elements=list(COUNTRIES.keys()))


def random_city(country):
    return fake.random_element(elements=COUNTRIES[country])


def generate_customer_data(n_customers: int = 15000) -> pd.DataFrame:

    rows = []

    segments = [
        "Premium",
        "Regular",
        "New",
        "At Risk",
    ]

    channels = [
        "Website",
        "Mobile App",
        "Retail Store",
        "Marketplace",
    ]

    income_groups = [
        "Low",
        "Middle",
        "High",
        "Ultra High",
    ]

    for idx in range(1, n_customers + 1):

        country = random_country()

        rows.append(
            {
                "customer_id": f"CUST-{idx:06d}",
                "customer_name": fake.name(),
                "age": fake.random_int(
                    min=18,
                    max=75,
                ),
                "gender": fake.random_element(
                    elements=[
                        "Male",
                        "Female",
                    ]
                ),
                "country": country,
                "city": random_city(country),
                "income_group": fake.random_element(elements=income_groups),
                "occupation": fake.job(),
                "signup_date": random_date(),
                "customer_segment": fake.random_element(elements=segments),
                "loyalty_score": fake.random_int(
                    min=1,
                    max=100,
                ),
                "preferred_channel": fake.random_element(elements=channels),
                "preferred_category": fake.random_element(
                    elements=list(PRODUCT_CATEGORIES.keys())
                ),
            }
        )

    return pd.DataFrame(rows)


def generate_product_data(n_products: int = 1000) -> pd.DataFrame:

    rows = []

    brands = [
        "NovaTech",
        "UrbanStyle",
        "HomeSphere",
        "ProGear",
        "LuxeCare",
        "Vertex",
        "Apex",
        "GlobalOne",
    ]

    suppliers = [
        "United States",
        "Germany",
        "United Kingdom",
        "France",
        "China",
        "Italy",
        "Netherlands",
    ]

    for idx in range(1, n_products + 1):

        category = fake.random_element(elements=list(PRODUCT_CATEGORIES.keys()))

        subcategory = fake.random_element(elements=PRODUCT_CATEGORIES[category])

        cost_price = round(
            np.random.uniform(5, 1500),
            2,
        )

        selling_price = round(
            cost_price * np.random.uniform(1.2, 2.8),
            2,
        )

        rows.append(
            {
                "product_id": f"PROD-{idx:05d}",
                "product_name": (
                    fake.word().capitalize() + " " + fake.word().capitalize()
                ),
                "brand": fake.random_element(elements=brands),
                "category": category,
                "subcategory": subcategory,
                "supplier_country": fake.random_element(elements=suppliers),
                "cost_price": cost_price,
                "selling_price": selling_price,
                "profit_margin": round(
                    ((selling_price - cost_price) / selling_price) * 100,
                    2,
                ),
                "product_rating": round(
                    np.random.uniform(3, 5),
                    1,
                ),
                "inventory_level": fake.random_int(
                    min=20,
                    max=2000,
                ),
                "launch_date": random_date(),
            }
        )

    return pd.DataFrame(rows)


def generate_transaction_data(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    n_transactions: int = 120000,
) -> pd.DataFrame:

    rows = []

    customer_ids = customers["customer_id"].tolist()

    product_lookup = products.set_index("product_id").to_dict("index")

    product_ids = list(product_lookup.keys())

    sales_channels = [
        "Website",
        "Mobile App",
        "Retail Store",
        "Marketplace",
    ]

    payment_methods = [
        "Credit Card",
        "Debit Card",
        "PayPal",
        "Apple Pay",
        "Google Pay",
    ]

    order_statuses = [
        "Completed",
        "Completed",
        "Completed",
        "Returned",
        "Cancelled",
    ]

    customer_lookup = customers.set_index("customer_id").to_dict("index")

    for idx in range(1, n_transactions + 1):

        customer_id = fake.random_element(elements=customer_ids)

        product_id = fake.random_element(elements=product_ids)

        customer = customer_lookup[customer_id]
        product = product_lookup[product_id]

        country = customer["country"]

        currency, exchange_rate = CURRENCIES[country]

        quantity = fake.random_int(
            min=1,
            max=10,
        )

        unit_price = float(product["selling_price"])

        gross_sales = round(
            quantity * unit_price,
            2,
        )

        discount_percentage = np.random.choice(
            [0, 0.05, 0.10, 0.15, 0.20],
            p=[0.35, 0.25, 0.20, 0.15, 0.05],
        )

        discount_amount = round(
            gross_sales * discount_percentage,
            2,
        )

        net_sales = round(
            gross_sales - discount_amount,
            2,
        )

        cost = round(
            quantity * float(product["cost_price"]),
            2,
        )

        profit = round(
            net_sales - cost,
            2,
        )

        rows.append(
            {
                "transaction_id": f"TXN-{idx:07d}",
                "customer_id": customer_id,
                "product_id": product_id,
                "order_date": random_date(),
                "country": country,
                "city": customer["city"],
                "currency": currency,
                "exchange_rate_to_usd": exchange_rate,
                "quantity": quantity,
                "unit_price": unit_price,
                "gross_sales": gross_sales,
                "discount_percentage": discount_percentage,
                "discount_amount": discount_amount,
                "net_sales": net_sales,
                "sales_usd": round(
                    net_sales * exchange_rate,
                    2,
                ),
                "cost": cost,
                "profit": profit,
                "profit_margin": round(
                    (profit / net_sales) * 100 if net_sales > 0 else 0,
                    2,
                ),
                "sales_channel": fake.random_element(elements=sales_channels),
                "payment_method": fake.random_element(elements=payment_methods),
                "order_status": fake.random_element(elements=order_statuses),
            }
        )

    return pd.DataFrame(rows)


def generate_marketing_data(
    customers: pd.DataFrame,
    n_campaigns: int = 500,
) -> pd.DataFrame:

    rows = []

    channels = [
        "Email",
        "Google Ads",
        "Facebook",
        "Instagram",
        "LinkedIn",
    ]

    customer_ids = customers["customer_id"].tolist()

    for idx in range(1, n_campaigns + 1):

        selected = np.random.choice(
            customer_ids,
            size=np.random.randint(100, 500),
            replace=False,
        )

        for customer_id in selected:

            impressions = fake.random_int(
                min=100,
                max=50000,
            )

            clicks = fake.random_int(
                min=10,
                max=5000,
            )

            rows.append(
                {
                    "campaign_id": f"CAMP-{idx:05d}",
                    "customer_id": customer_id,
                    "campaign_date": random_date(),
                    "channel": fake.random_element(elements=channels),
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": fake.random_int(
                        min=0,
                        max=100,
                    ),
                    "campaign_cost": round(
                        np.random.uniform(
                            500,
                            50000,
                        ),
                        2,
                    ),
                    "engagement_score": fake.random_int(
                        min=1,
                        max=100,
                    ),
                }
            )

    return pd.DataFrame(rows)


def generate_support_data(
    customers: pd.DataFrame,
    n_tickets: int = 5000,
) -> pd.DataFrame:

    rows = []

    issues = [
        "Billing",
        "Delivery",
        "Technical",
        "Refund",
        "Product Information",
    ]

    customer_ids = customers["customer_id"].tolist()

    for idx in range(1, n_tickets + 1):

        rows.append(
            {
                "ticket_id": f"TICKET-{idx:06d}",
                "customer_id": fake.random_element(elements=customer_ids),
                "created_date": random_date(),
                "issue_type": fake.random_element(elements=issues),
                "resolution_hours": fake.random_int(
                    min=1,
                    max=240,
                ),
                "customer_satisfaction": fake.random_int(
                    min=1,
                    max=5,
                ),
                "resolved": fake.boolean(chance_of_getting_true=90),
            }
        )

    return pd.DataFrame(rows)


def generate_inventory_data(
    products: pd.DataFrame,
    n_records: int = 10000,
) -> pd.DataFrame:

    rows = []

    warehouses = [
        "New York",
        "California",
        "London",
        "Berlin",
        "Paris",
        "Amsterdam",
    ]

    product_ids = products["product_id"].tolist()

    for idx in range(1, n_records + 1):

        stock = fake.random_int(
            min=0,
            max=5000,
        )

        reorder_point = fake.random_int(
            min=50,
            max=500,
        )

        rows.append(
            {
                "inventory_id": f"INV-{idx:06d}",
                "product_id": fake.random_element(elements=product_ids),
                "warehouse": fake.random_element(elements=warehouses),
                "inventory_date": random_date(),
                "stock_on_hand": stock,
                "reorder_point": reorder_point,
                "stock_risk": ("High" if stock < reorder_point else "Normal"),
            }
        )

    return pd.DataFrame(rows)


def generate_all_data() -> Dict[str, pd.DataFrame]:

    print("Generating customers...")
    customers = generate_customer_data()

    print("Generating products...")
    products = generate_product_data()

    print("Generating transactions...")
    transactions = generate_transaction_data(
        customers,
        products,
    )

    print("Generating marketing data...")
    marketing = generate_marketing_data(customers)

    print("Generating support data...")
    support = generate_support_data(customers)

    print("Generating inventory data...")
    inventory = generate_inventory_data(products)

    return {
        "customers": customers,
        "products": products,
        "transactions": transactions,
        "marketing": marketing,
        "support": support,
        "inventory": inventory,
    }


def save_all_data(output_dir: str | Path = OUTPUT_DIR):

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = generate_all_data()

    for name, dataframe in datasets.items():

        file_path = output_path / f"{name}.csv"

        dataframe.to_csv(
            file_path,
            index=False,
        )

        print(f"Saved {file_path}: {len(dataframe):,} rows")


if __name__ == "__main__":

    save_all_data()
