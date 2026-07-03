"""
ml_models.py
============
Smart Bakery ML Models — trained on real Bakery.csv data.

Models:
  1. Linear Regression  — features: day, month, day_of_week
  2. Random Forest      — features: day, month, day_of_week, is_weekend, is_holiday, year

Targets:
  - total_items        (total items sold per day)
  - total_transactions (unique transactions per day)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from data_loader import (
    load_dataset, get_feature_matrix, is_holiday,
    _item_col, DAY_NAMES, HOLIDAY_DATES
)

# Feature sets
LR_FEATURES = ["day", "month", "day_of_week"]
RF_FEATURES  = ["day", "month", "day_of_week", "is_weekend", "is_holiday",
                "year", "day_of_year", "week_of_year", "quarter"]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        "model":    name,
        "mae":      round(float(mae),  2),
        "rmse":     round(float(rmse), 2),
        "r2":       round(float(r2),   4),
        "accuracy": round(max(0.0, float(r2)) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class BakeryMLModels:
    """
    Trains Linear Regression and Random Forest on the aggregated daily data
    derived from Bakery.csv.
    """

    def __init__(self):
        self.raw_df    = None
        self.daily_df  = None
        self.top_items = []

        # Models keyed by target name
        self.lr_models: dict[str, LinearRegression]      = {}
        self.rf_models: dict[str, RandomForestRegressor] = {}

        self.lr_scaler = StandardScaler()
        self.metrics: dict  = {}
        self._trained: bool = False

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, csv_path: str = None) -> dict:
        """
        Load CSV data, aggregate, train both models on every target.
        Returns full metrics dict.
        """
        if csv_path is None:
            from data_loader import CSV_PATH
            csv_path = CSV_PATH
        print(f"[ML] Loading {csv_path} ...")
        self.raw_df, self.daily_df, self.top_items = load_dataset(csv_path)
        df = self.daily_df

        targets = ["total_items", "total_transactions"]
        metrics_out = {}

        # Fit scaler once on LR features
        X_lr_full = df[LR_FEATURES].values
        self.lr_scaler.fit(X_lr_full)

        for target in targets:
            y = df[target].values

            # ── Linear Regression ──────────────────────────────────────
            X_lr = self.lr_scaler.transform(df[LR_FEATURES].values)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_lr, y, test_size=0.2, random_state=42)

            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            self.lr_models[target] = lr
            lr_pred = lr.predict(X_te)

            # ── Random Forest ──────────────────────────────────────────
            X_rf = df[RF_FEATURES].values
            X_tr_rf, X_te_rf, _, _ = train_test_split(
                X_rf, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_tr_rf, y_tr)
            self.rf_models[target] = rf
            rf_pred = rf.predict(X_te_rf)

            metrics_out[target] = {
                "linear_regression": _metrics(y_te, lr_pred, "Linear Regression"),
                "random_forest":     _metrics(y_te, rf_pred, "Random Forest"),
            }
            print(f"[ML] {target}: LR R²={metrics_out[target]['linear_regression']['r2']:.4f} "
                  f"| RF R²={metrics_out[target]['random_forest']['r2']:.4f}")

        self.metrics  = metrics_out
        self._trained = True
        return metrics_out

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, dates: list[dict], target: str = "total_items") -> dict:
        """
        Predict for a list of date dicts.

        Each dict must have: day, month, year, day_of_week, is_weekend, is_holiday

        Returns {"linear_regression": [...], "random_forest": [...]}
        """
        if not self._trained:
            raise RuntimeError("Call .train() first.")

        lr_rows, rf_rows = [], []
        for d in dates:
            lr_rows.append([d["day"], d["month"], d["day_of_week"]])
            rf_rows.append([d["day"], d["month"], d["day_of_week"],
                            d["is_weekend"], d["is_holiday"], d["year"],
                            d["day_of_year"], d["week_of_year"], d["quarter"]])

        X_lr = self.lr_scaler.transform(np.array(lr_rows))
        X_rf = np.array(rf_rows)

        lr_p = np.maximum(0, self.lr_models[target].predict(X_lr))
        rf_p = np.maximum(0, self.rf_models[target].predict(X_rf))

        return {
            "linear_regression": [round(float(v), 1) for v in lr_p],
            "random_forest":     [round(float(v), 1) for v in rf_p],
        }

    # ------------------------------------------------------------------ #
    #  Feature Importance                                                 #
    # ------------------------------------------------------------------ #
    def feature_importance(self, target: str = "total_items") -> dict:
        if not self._trained:
            raise RuntimeError("Call .train() first.")
        imp = self.rf_models[target].feature_importances_
        return {feat: round(float(imp[i]), 4) for i, feat in enumerate(RF_FEATURES)}

    # ------------------------------------------------------------------ #
    #  Dashboard data helpers                                             #
    # ------------------------------------------------------------------ #
    def get_monthly_summary(self) -> list[dict]:
        df = self.daily_df.copy()
        monthly = (
            df.groupby(["year", "month"])
              .agg(total_items=("total_items", "sum"),
                   total_transactions=("total_transactions", "sum"))
              .reset_index()
        )
        monthly["label"] = monthly.apply(
            lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1)
        return monthly.to_dict(orient="records")

    def get_weekly_average(self) -> list[dict]:
        df  = self.daily_df.copy()
        avg = df.groupby("day_of_week")["total_items"].mean().reset_index()
        avg["label"] = avg["day_of_week"].apply(lambda x: DAY_NAMES[int(x)])
        return avg.to_dict(orient="records")

    def get_top_items(self, n: int = 10) -> dict:
        """Top N items by total count from raw data."""
        counts = self.raw_df["Items"].value_counts().head(n)
        return counts.to_dict()

    def get_daypart_breakdown(self) -> dict:
        """Sales breakdown by time of day."""
        df = self.raw_df.copy()
        df["Daypart"] = df["Daypart"].str.strip().str.title()
        counts = df["Daypart"].value_counts()
        return counts.to_dict()

    def get_last_n_days(self, n: int = 30) -> list[dict]:
        return (
            self.daily_df.tail(n)
                .assign(date=lambda d: d["date_str"])
                [["date", "total_items", "total_transactions",
                  "is_weekend", "is_holiday"]]
                .to_dict(orient="records")
        )

    def get_summary_stats(self) -> dict:
        df = self.daily_df
        return {
            "total_records":           len(self.raw_df),
            "total_days":              len(df),
            "date_range_start":        df["date_str"].min(),
            "date_range_end":          df["date_str"].max(),
            "avg_items_per_day":       round(float(df["total_items"].mean()), 1),
            "avg_transactions_per_day":round(float(df["total_transactions"].mean()), 1),
            "max_items_day":           int(df["total_items"].max()),
            "weekend_avg":             round(float(df[df["is_weekend"]==1]["total_items"].mean()), 1),
            "weekday_avg":             round(float(df[df["is_weekend"]==0]["total_items"].mean()), 1),
            "top_items":               self.get_top_items(5),
        }

    # ------------------------------------------------------------------ #
    #  Inventory alerts                                                   #
    # ------------------------------------------------------------------ #
    def inventory_alerts(self, stock: dict, forecast_days: int = 7) -> list[dict]:
        """
        Predict next *forecast_days* total demand and raise alerts for low stock.
        Uses top-5 items as products.
        """
        from datetime import datetime, timedelta

        today = datetime.today()
        dates = []
        for i in range(forecast_days):
            dt  = today + timedelta(days=i)
            dow = dt.weekday()
            dates.append({
                "day": dt.day, "month": dt.month, "year": dt.year,
                "day_of_week": dow,
                "is_weekend":  int(dow >= 5),
                "is_holiday":  is_holiday(dt),
                "day_of_year": dt.timetuple().tm_yday,
                "week_of_year": int(dt.strftime("%W")),
                "quarter":     (dt.month - 1) // 3 + 1,
            })

        pred = self.predict(dates, "total_items")
        total_demand = sum(pred["random_forest"])

        # Distribute demand proportionally across top items
        top5  = self.get_top_items(5)
        grand = sum(top5.values()) or 1
        alerts = []
        for item, hist_count in top5.items():
            proportion = hist_count / grand
            demand     = round(total_demand * proportion, 0)
            avail      = stock.get(item, 0)
            status     = "critical" if avail < demand * 0.5 \
                         else "low"  if avail < demand \
                         else "ok"
            alerts.append({
                "product":          item,
                "current_stock":    avail,
                "predicted_demand": int(demand),
                "status":           status,
                "days_coverage":    round(avail / (demand / forecast_days), 1)
                                    if demand > 0 else 999,
            })
        return alerts

    # ------------------------------------------------------------------ #
    #  Customer Segmentation (K-Means)                                    #
    # ------------------------------------------------------------------ #
    def get_customer_segments(self) -> dict:
        """
        Group transactions by TransactionNo, build simple features,
        and run KMeans to segment customers.
        """
        if not self._trained:
            raise RuntimeError("Call .train() first.")
        
        df = self.raw_df.copy()
        
        # Extract transaction-level features
        tx = df.groupby("TransactionNo").agg(
            total_items=("Items", "count"),
            datetime=("DateTime", "first"),
            daypart=("Daypart", "first"),
            daytype=("DayType", "first")
        ).reset_index()
        
        tx["hour"] = tx["datetime"].dt.hour
        tx["is_weekend"] = (tx["daytype"] == "Weekend").astype(int)
        
        # Count main item categories per transaction to see preference
        item_counts = pd.crosstab(df["TransactionNo"], df["Items"])
        for item in ["Coffee", "Bread", "Tea", "Cake", "Pastry"]:
            if item in item_counts.columns:
                tx[f"has_{item.lower()}"] = tx["TransactionNo"].map(item_counts[item] > 0).astype(int)
            else:
                tx[f"has_{item.lower()}"] = 0
            
        # Build features matrix
        features = ["hour", "is_weekend", "total_items", "has_coffee", "has_bread", "has_tea", "has_cake", "has_pastry"]
        X = tx[features].fillna(0).values
        
        # Run KMeans with 3 clusters
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        tx["cluster"] = labels
        
        cluster_summary = []
        for i in range(3):
            subset = tx[tx["cluster"] == i]
            avg_items = float(subset["total_items"].mean())
            avg_hour = float(subset["hour"].mean())
            coffee_pct = float(subset["has_coffee"].mean())
            bread_pct = float(subset["has_bread"].mean())
            cake_pct = float(subset["has_cake"].mean())
            
            # Label based on cluster profile
            if coffee_pct > 0.49 and avg_items < 2.2:
                label = "Coffee & Beverage Lovers"
            elif cake_pct > 0.12:
                label = "Dessert & Pastry Indulgers"
            else:
                label = "Family & Bread Shoppers"
            
            cluster_summary.append({
                "cluster_id": i,
                "label": label,
                "count": int(len(subset)),
                "avg_items": round(avg_items, 2),
                "avg_hour": f"{int(avg_hour):02d}:00",
                "coffee_pct": round(coffee_pct * 100, 1),
                "bread_pct": round(bread_pct * 100, 1),
                "cake_pct": round(cake_pct * 100, 1),
            })
            
        # Ensure distinct labels
        seen_labels = {}
        for c in cluster_summary:
            lbl = c["label"]
            if lbl in seen_labels:
                seen_labels[lbl] += 1
                c["label"] = f"{lbl} Group {seen_labels[lbl]}"
            else:
                seen_labels[lbl] = 1
                
        return {"clusters": cluster_summary}

    # ------------------------------------------------------------------ #
    #  Combo Recommendation (Apriori-style co-occurrences)               #
    # ------------------------------------------------------------------ #
    def get_combo_recommendations(self) -> list[dict]:
        """
        Calculates top combos frequently purchased together in the same transaction.
        """
        if not self._trained:
            raise RuntimeError("Call .train() first.")
            
        df = self.raw_df.copy()
        
        # Group items per transaction
        tx_items = df.groupby("TransactionNo")["Items"].apply(list).to_dict()
        total_tx = len(tx_items)
        if total_tx == 0:
            return []
            
        item_counts = df["Items"].value_counts().to_dict()
        
        from collections import defaultdict
        pair_counts = defaultdict(int)
        
        for items in tx_items.values():
            unique_items = sorted(list(set(items)))
            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    pair_counts[(unique_items[i], unique_items[j])] += 1
                    
        combos = []
        for (item1, item2), count in pair_counts.items():
            support = count / total_tx
            conf1 = count / item_counts.get(item1, 1)
            conf2 = count / item_counts.get(item2, 1)
            
            combos.append({
                "item1": item1,
                "item2": item2,
                "support": round(support * 100, 2),
                "confidence_1_to_2": round(conf1 * 100, 1),
                "confidence_2_to_1": round(conf2 * 100, 1),
                "count": count
            })
            
        combos = sorted(combos, key=lambda x: x["count"], reverse=True)[:5]
        return combos

    # ------------------------------------------------------------------ #
    #  Wastage Prediction & Optimization                                  #
    # ------------------------------------------------------------------ #
    def get_wastage_report(self, stock: dict, current_hour: int = 15) -> list[dict]:
        """
        Predict end-of-day wastage and suggest discount timing.
        """
        if not self._trained:
            raise RuntimeError("Call .train() first.")
            
        if current_hour < 12:
            fraction_completed = 0.25
        elif current_hour < 17:
            fraction_completed = 0.75
        elif current_hour < 20:
            fraction_completed = 0.95
        else:
            fraction_completed = 1.0
            
        from datetime import datetime
        today = datetime.today()
        dow = today.weekday()
        date_input = [{
            "day": today.day, "month": today.month, "year": today.year,
            "day_of_week": dow, "is_weekend": int(dow >= 5),
            "is_holiday": is_holiday(today),
            "day_of_year":  today.timetuple().tm_yday,
            "week_of_year": int(today.strftime("%W")),
            "quarter":      (today.month - 1) // 3 + 1,
        }]
        
        pred_rf = self.predict(date_input, "total_items")["random_forest"][0]
        avg_daily = self.daily_df["total_items"].mean() or 1.0
        demand_factor = pred_rf / avg_daily
        
        top5 = self.get_top_items(5)
        grand = sum(top5.values()) or 1
        
        report = []
        for item, hist_count in top5.items():
            item_daily_avg = (hist_count / len(self.daily_df))
            expected_today_demand = max(1.0, item_daily_avg * demand_factor)
            expected_sold = expected_today_demand * fraction_completed
            avail = stock.get(item, 0)
            remaining_expected_sales = expected_today_demand * (1.0 - fraction_completed)
            predicted_waste = max(0.0, avail - remaining_expected_sales)
            
            waste_pct = (predicted_waste / avail * 100) if avail > 0 else 0
            if predicted_waste > 2 and waste_pct > 15:
                discount_pct = 30 if waste_pct < 40 else 50
                rec_hour = min(18, max(current_hour + 1, 16))
                discount_action = f"Reduce price by {discount_pct}% at {rec_hour:02d}:00 PM"
            else:
                discount_action = "Maintain price"
                
            report.append({
                "product": item,
                "current_stock": avail,
                "expected_demand_today": int(expected_today_demand),
                "expected_sold_so_far": int(expected_sold),
                "predicted_waste": int(predicted_waste),
                "waste_percentage": round(waste_pct, 1),
                "discount_action": discount_action
            })
        return report

    # ------------------------------------------------------------------ #
    #  Sales Logging & CSV Persistence                                    #
    # ------------------------------------------------------------------ #
    def record_sale(self, items_list: list[str], csv_path: str = None) -> bool:
        """
        Record sale of items list, write back to CSV, and retrain models.
        """
        import os
        import csv
        from datetime import datetime

        if not self._trained:
            raise RuntimeError("Call .train() first.")
            
        if csv_path is None:
            from data_loader import CSV_PATH
            csv_path = CSV_PATH
            
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        hour = now.hour
        if 6 <= hour < 12:
            daypart = "Morning"
        elif 12 <= hour < 17:
            daypart = "Afternoon"
        elif 17 <= hour < 21:
            daypart = "Evening"
        else:
            daypart = "Night"
            
        daytype = "Weekend" if now.weekday() >= 5 else "Weekday"
        next_tx_no = int(self.raw_df["TransactionNo"].max() + 1)
        
        new_rows = []
        for item in items_list:
            new_rows.append({
                "TransactionNo": next_tx_no,
                "Items": item,
                "DateTime": now,
                "Daypart": daypart,
                "DayType": daytype
            })
            
        new_df = pd.DataFrame(new_rows)
        self.raw_df = pd.concat([self.raw_df, new_df], ignore_index=True)
        
        file_exists = os.path.exists(csv_path)
        csv_rows = []
        for r in new_rows:
            csv_rows.append([
                r["TransactionNo"],
                r["Items"],
                now_str,
                r["Daypart"],
                r["DayType"]
            ])
            
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writerow(["TransactionNo", "Items", "DateTime", "Daypart", "DayType"])
            writer.writerows(csv_rows)
            
        print(f"[ML] POS transaction recorded. Appended {len(items_list)} items to {csv_path}")
        self.train(csv_path)
        return True


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bm = BakeryMLModels()
    bm.train()

    print("\n=== Summary Stats ===")
    for k, v in bm.get_summary_stats().items():
        print(f"  {k}: {v}")

    print("\n=== Feature Importance (RF / total_items) ===")
    for f, imp in bm.feature_importance().items():
        print(f"  {f}: {imp:.4f}")

    print("\n=== 3-day prediction ===")
    from datetime import datetime, timedelta
    dates = []
    for i in range(3):
        dt  = datetime.today() + timedelta(days=i)
        dow = dt.weekday()
        dates.append({
            "day": dt.day, "month": dt.month, "year": dt.year,
            "day_of_week": dow,
            "is_weekend": int(dow >= 5),
            "is_holiday": is_holiday(dt),
            "day_of_year":  dt.timetuple().tm_yday,
            "week_of_year": int(dt.strftime("%W")),
            "quarter":      (dt.month - 1) // 3 + 1,
        })
    p = bm.predict(dates)
    print("  LR:", p["linear_regression"])
    print("  RF:", p["random_forest"])
