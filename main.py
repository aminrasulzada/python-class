"""
Cafe sales project: reading, cleaning and statistics for the Dirty Cafe Sales dataset.
Dataset: https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/dirty_cafe_sales.csv")
CLEAN_PATH = Path("data/clean/cafe_sales_clean.csv")
RESULTS_DIR = Path("results")

MENU = {
    "Coffee": 2.0, "Tea": 1.5, "Sandwich": 4.0, "Salad": 5.0,
    "Cake": 3.0, "Cookie": 1.0, "Smoothie": 4.0, "Juice": 3.0,
}


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and show a quick overview."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
    return df


def missing_report(df: pd.DataFrame, title: str) -> None:
    """Print missing values per column."""
    print(f"\n=== {title} ===")
    print(df.isna().sum().to_string())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw cafe sales data and return a new DataFrame."""
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df = df.replace(["ERROR", "UNKNOWN"], np.nan)
    missing_report(df, "Missing values after marking ERROR/UNKNOWN")

    df = df.drop_duplicates(subset="transaction_id")

    for col in ["quantity", "price_per_unit", "total_spent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    df["price_per_unit"] = df["price_per_unit"].fillna(df["item"].map(MENU))

    df["price_per_unit"] = df["price_per_unit"].fillna(df["total_spent"] / df["quantity"])
    df["quantity"] = df["quantity"].fillna(df["total_spent"] / df["price_per_unit"])
    df["total_spent"] = df["total_spent"].fillna(df["quantity"] * df["price_per_unit"])

    price_counts = pd.Series(MENU).value_counts()
    unique_price_to_item = {
        price: item for item, price in MENU.items() if price_counts[price] == 1
    }
    df["item"] = df["item"].fillna(df["price_per_unit"].map(unique_price_to_item))

    before = len(df)
    df = df.dropna(subset=["item", "quantity", "price_per_unit", "total_spent", "transaction_date"])
    print(f"\nDropped {before - len(df)} rows that could not be recovered")

    df["quantity"] = df["quantity"].round().astype(int)

    for col in ["payment_method", "location"]:
        df[col] = df[col].fillna("Unknown")

    missing_report(df, "Missing values after cleaning")
    return df.reset_index(drop=True)


def save_data(df: pd.DataFrame, path: Path = CLEAN_PATH) -> None:
    """Save the cleaned data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} clean rows to {path}")


# ---------- Statistics ----------

def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Add month and weekday columns used by the statistics."""
    df = df.copy()
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
    df["weekday"] = df["transaction_date"].dt.day_name()
    return df


def overall_summary(df: pd.DataFrame) -> pd.Series:
    """Key headline numbers for the whole dataset."""
    return pd.Series({
        "transactions": len(df),
        "total_revenue": df["total_spent"].sum(),
        "items_sold": df["quantity"].sum(),
        "avg_transaction_value": df["total_spent"].mean(),
        "median_transaction_value": df["total_spent"].median(),
        "first_date": df["transaction_date"].min().date(),
        "last_date": df["transaction_date"].max().date(),
    })


def group_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Transactions, items sold, revenue and revenue share for each group."""
    stats = df.groupby(column).agg(
        transactions=("transaction_id", "count"),
        items_sold=("quantity", "sum"),
        revenue=("total_spent", "sum"),
        avg_transaction=("total_spent", "mean"),
    )
    stats["revenue_share_%"] = 100 * stats["revenue"] / stats["revenue"].sum()
    return stats.sort_values("revenue", ascending=False).round(2)


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and transactions per month, in date order."""
    return group_stats(df, "month").sort_index()


def weekday_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and transactions per day of the week, Monday to Sunday."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return group_stats(df, "weekday").reindex(order)


def print_table(title: str, table) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string())


def run_statistics(df: pd.DataFrame) -> None:
    """Print all statistics and save them as CSV files in the results folder."""
    df = add_date_parts(df)
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = overall_summary(df)
    print_table("Overall summary", summary)
    print_table("Numeric columns", df[["quantity", "price_per_unit", "total_spent"]].describe().round(2))

    tables = {
        "by_item": group_stats(df, "item"),
        "by_payment_method": group_stats(df, "payment_method"),
        "by_location": group_stats(df, "location"),
        "by_month": monthly_stats(df),
        "by_weekday": weekday_stats(df),
    }
    for name, table in tables.items():
        print_table(name.replace("_", " ").title(), table)
        table.to_csv(RESULTS_DIR / f"stats_{name}.csv")

    summary.to_csv(RESULTS_DIR / "stats_overall.csv", header=["value"])
    print(f"\nSaved all statistics tables to the '{RESULTS_DIR}' folder")


if __name__ == "__main__":
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    save_data(clean_df)

    run_statistics(clean_df)