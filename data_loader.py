"""
data_loader.py
==============
Loads and preprocesses the real Bakery.csv (transaction-level data) and
transforms it into a daily-aggregated DataFrame suitable for ML training.

CSV columns: TransactionNo, Items, DateTime, Daypart, DayType
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path to CSV (same folder as this script)
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Bakery.csv")

# ---------------------------------------------------------------------------
# Holiday dates (MM-DD) used for feature engineering
# ---------------------------------------------------------------------------
HOLIDAY_DATES = {
    "01-01", "02-14", "03-08", "04-20", "05-01",
    "06-21", "10-31", "11-25", "12-24", "12-25", "12-31",
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]


def is_holiday(dt) -> int:
    """Return 1 if dt (datetime or date) is a recognised holiday."""
    return int(pd.Timestamp(dt).strftime("%m-%d") in HOLIDAY_DATES)


# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------
def load_raw(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Parse DateTime
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime", "Items"])

    # Normalise item names
    df["Items"] = df["Items"].str.strip().str.title()

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into one row per day.

    Returns a DataFrame with columns:
        date, day, month, year, day_of_week, is_weekend, is_holiday,
        daypart_morning, daypart_afternoon, daypart_evening, daypart_night,
        total_transactions, total_items,
        <top-item>_count  (×N top items)
    """
    df = df.copy()
    df["date"] = df["DateTime"].dt.normalize()

    # ── Daily item counts per top product ──────────────────────────────────
    top_items = (
        df["Items"].value_counts().head(10).index.tolist()
    )
    for item in top_items:
        col = _item_col(item)
        df[col] = (df["Items"] == item).astype(int)

    # ── Daypart dummies ────────────────────────────────────────────────────
    df["Daypart"] = df["Daypart"].str.strip().str.title()
    for part in ["Morning", "Afternoon", "Evening", "Night"]:
        df[f"daypart_{part.lower()}"] = (df["Daypart"] == part).astype(int)

    # ── Group by date ──────────────────────────────────────────────────────
    agg_dict = {
        "TransactionNo": pd.Series.nunique,  # unique transactions per day
        "Items":         "count",            # total items sold
    }
    for item in top_items:
        agg_dict[_item_col(item)] = "sum"
    for part in ["Morning", "Afternoon", "Evening", "Night"]:
        agg_dict[f"daypart_{part.lower()}"] = "sum"

    daily = df.groupby("date").agg(agg_dict).reset_index()
    daily.rename(columns={
        "TransactionNo": "total_transactions",
        "Items":         "total_items",
    }, inplace=True)

    # ── Date features ──────────────────────────────────────────────────────
    daily["day"]         = daily["date"].dt.day
    daily["month"]       = daily["date"].dt.month
    daily["year"]        = daily["date"].dt.year
    daily["day_of_week"] = daily["date"].dt.dayofweek      # 0=Mon … 6=Sun
    daily["is_weekend"]  = (daily["day_of_week"] >= 5).astype(int)
    daily["is_holiday"]  = daily["date"].apply(is_holiday)
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["week_of_year"]= daily["date"].dt.isocalendar().week.astype(int)
    daily["quarter"]     = daily["date"].dt.quarter
    daily["date_str"]    = daily["date"].dt.strftime("%Y-%m-%d")

    # Sort chronologically
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily, top_items


def _item_col(item: str) -> str:
    """Convert item name to a safe column name."""
    return item.lower().replace(" ", "_").replace("-", "_") + "_count"


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------
def get_feature_matrix(daily: pd.DataFrame, extended: bool = True):
    """
    Build X and y for model training.

    Parameters
    ----------
    daily    : output of aggregate_daily()
    extended : if True  → LR features + is_weekend, is_holiday, year (for RF)
               if False → day, month, day_of_week only              (for LR)

    Returns
    -------
    X         : np.ndarray
    y_dict    : {"total_items": arr, "total_transactions": arr}
    features  : list[str]  (column names for X)
    """
    base_features = ["day", "month", "day_of_week"]
    ext_features  = base_features + ["is_weekend", "is_holiday", "year",
                                     "day_of_year", "week_of_year", "quarter"]

    features = ext_features if extended else base_features
    X = daily[features].values

    y_dict = {
        "total_items":        daily["total_items"].values,
        "total_transactions": daily["total_transactions"].values,
    }
    return X, y_dict, features


# ---------------------------------------------------------------------------
# Convenience: load everything in one call
# ---------------------------------------------------------------------------
def load_dataset(csv_path: str = CSV_PATH):
    """
    Load Bakery.csv → aggregate → return (raw_df, daily_df, top_items).
    """
    raw   = load_raw(csv_path)
    daily, top_items = aggregate_daily(raw)
    return raw, daily, top_items


# ---------------------------------------------------------------------------
# Quick inspection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw, daily, top_items = load_dataset()
    print("Raw transactions :", len(raw))
    print("Daily records    :", len(daily))
    print("Date range       :", daily["date_str"].min(), "→", daily["date_str"].max())
    print("Top items        :", top_items)
    print(daily.head(3).to_string())
