"""
app.py
======
Flask web server for the Smart Bakery Management System.
Reads real Bakery.csv data, trains both ML models on startup,
and serves a REST API consumed by the premium dashboard.
"""

from flask import Flask, jsonify, request, render_template
from datetime import datetime, timedelta

from data_loader import is_holiday
from ml_models import BakeryMLModels

import os
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
app = Flask(__name__)
bm  = BakeryMLModels()

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ACTIVE_DATASET = "Default (Bakery.csv)"

# Default stock levels — user can update via POST /api/inventory
CURRENT_STOCK = {
    "Coffee":       500,
    "Bread":        350,
    "Tea":          400,
    "Cake":         120,
    "Pastry":       280,
}


@app.before_request
def _ensure_trained():
    if not bm._trained:
        bm.train()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API — model metrics
# ---------------------------------------------------------------------------
@app.route("/api/metrics")
def api_metrics():
    return jsonify(bm.metrics)


# ---------------------------------------------------------------------------
# API — historical data
# ---------------------------------------------------------------------------
@app.route("/api/monthly")
def api_monthly():
    return jsonify(bm.get_monthly_summary())


@app.route("/api/weekly")
def api_weekly():
    return jsonify(bm.get_weekly_average())


@app.route("/api/top-items")
def api_top_items():
    n = int(request.args.get("n", 10))
    return jsonify(bm.get_top_items(n))


@app.route("/api/daypart")
def api_daypart():
    return jsonify(bm.get_daypart_breakdown())


@app.route("/api/recent")
def api_recent():
    n = int(request.args.get("n", 30))
    return jsonify(bm.get_last_n_days(n))


# ---------------------------------------------------------------------------
# API — predictions
# ---------------------------------------------------------------------------
@app.route("/api/predict")
def api_predict():
    """
    Query params:
      days   (int) : days ahead to predict    [default: 14]
      target (str) : total_items | total_transactions
    """
    days   = int(request.args.get("days", 14))
    target = request.args.get("target", "total_items")

    if target not in ("total_items", "total_transactions"):
        target = "total_items"

    today     = datetime.today()
    date_list = []
    labels    = []

    for i in range(days):
        dt  = today + timedelta(days=i)
        dow = dt.weekday()
        date_list.append({
            "day":          dt.day,
            "month":        dt.month,
            "year":         dt.year,
            "day_of_week":  dow,
            "is_weekend":   int(dow >= 5),
            "is_holiday":   is_holiday(dt),
            "day_of_year":  dt.timetuple().tm_yday,
            "week_of_year": int(dt.strftime("%W")),
            "quarter":      (dt.month - 1) // 3 + 1,
        })
        labels.append(dt.strftime("%b %d"))

    preds = bm.predict(date_list, target)
    return jsonify({"labels": labels, **preds})


# ---------------------------------------------------------------------------
# API — feature importance
# ---------------------------------------------------------------------------
@app.route("/api/feature-importance")
def api_feature_importance():
    target = request.args.get("target", "total_items")
    if target not in ("total_items", "total_transactions"):
        target = "total_items"
    return jsonify(bm.feature_importance(target))


# ---------------------------------------------------------------------------
# API — inventory
# ---------------------------------------------------------------------------
@app.route("/api/inventory", methods=["GET", "POST"])
def api_inventory():
    global CURRENT_STOCK
    if request.method == "POST":
        data = request.get_json(force=True)
        CURRENT_STOCK.update(data.get("stock", {}))

    alerts = bm.inventory_alerts(CURRENT_STOCK, forecast_days=7)
    return jsonify({"stock": CURRENT_STOCK, "alerts": alerts})


# ---------------------------------------------------------------------------
# API — summary (dashboard cards)
# ---------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    stats = bm.get_summary_stats()
    today = datetime.today()
    dow   = today.weekday()
    date_input = [{
        "day": today.day, "month": today.month, "year": today.year,
        "day_of_week": dow, "is_weekend": int(dow >= 5),
        "is_holiday": is_holiday(today),
        "day_of_year":  today.timetuple().tm_yday,
        "week_of_year": int(today.strftime("%W")),
        "quarter":      (today.month - 1) // 3 + 1,
    }]

    pred_lr = bm.predict(date_input, "total_items")["linear_regression"][0]
    pred_rf = bm.predict(date_input, "total_items")["random_forest"][0]

    return jsonify({
        "today":                    today.strftime("%A, %d %B %Y"),
        **stats,
        "predicted_today_lr":       pred_lr,
        "predicted_today_rf":       pred_rf,
        "active_dataset":           ACTIVE_DATASET,
    })


# ---------------------------------------------------------------------------
# API — upload custom CSV
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    global ACTIVE_DATASET
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # Retrain the models on the newly uploaded dataset
            bm.train(filepath)
            ACTIVE_DATASET = filename
            stats = bm.get_summary_stats()
            return jsonify({
                "message": "File uploaded and models retrained successfully.",
                "active_dataset": filename,
                "summary": stats
            })
        except Exception as e:
            # Revert to training default
            try:
                bm.train()
                ACTIVE_DATASET = "Default (Bakery.csv)"
            except Exception:
                pass
            return jsonify({"error": f"Failed to train models on the uploaded CSV: {str(e)}"}), 400
    return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400


# ---------------------------------------------------------------------------
# API — predict specific date
# ---------------------------------------------------------------------------
@app.route("/api/predict-date")
def api_predict_date():
    """
    Query params:
      date   (str) : YYYY-MM-DD
      target (str) : total_items | total_transactions
    """
    date_str = request.args.get("date")
    target   = request.args.get("target", "total_items")
    if target not in ("total_items", "total_transactions"):
        target = "total_items"
    
    if not date_str:
        return jsonify({"error": "Date parameter is required (YYYY-MM-DD)"}), 400
    
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
    dow = dt.weekday()
    date_input = [{
        "day": dt.day, "month": dt.month, "year": dt.year,
        "day_of_week": dow, "is_weekend": int(dow >= 5),
        "is_holiday": is_holiday(dt),
        "day_of_year": dt.timetuple().tm_yday,
        "week_of_year": int(dt.strftime("%W")),
        "quarter": (dt.month - 1) // 3 + 1,
    }]
    
    preds = bm.predict(date_input, target)
    return jsonify({
        "date": date_str,
        "is_weekend": int(dow >= 5),
        "is_holiday": is_holiday(dt),
        "linear_regression": preds["linear_regression"][0],
        "random_forest": preds["random_forest"][0]
    })


# ---------------------------------------------------------------------------
# API — customer insights (segmentation + combos)
# ---------------------------------------------------------------------------
@app.route("/api/customer-insights")
def api_customer_insights():
    try:
        segments = bm.get_customer_segments()
        combos = bm.get_combo_recommendations()
        return jsonify({
            "segments": segments["clusters"],
            "combos": combos
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API — wastage report & optimization
# ---------------------------------------------------------------------------
@app.route("/api/wastage")
def api_wastage():
    hour = int(request.args.get("hour", 15))
    try:
        report = bm.get_wastage_report(CURRENT_STOCK, hour)
        return jsonify({"hour": hour, "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API — POS recording
# ---------------------------------------------------------------------------
@app.route("/api/sales", methods=["POST"])
def api_sales():
    global CURRENT_STOCK
    data = request.get_json(force=True)
    items = data.get("items", [])
    if not items:
        return jsonify({"error": "No items provided"}), 400
        
    # Get active file path for saving
    from data_loader import CSV_PATH
    filepath = CSV_PATH
    if ACTIVE_DATASET != "Default (Bakery.csv)":
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], ACTIVE_DATASET)
        
    try:
        # Subtract stock levels
        for item in items:
            if item in CURRENT_STOCK:
                CURRENT_STOCK[item] = max(0, CURRENT_STOCK[item] - 1)
                
        # Record transaction to raw dataset and disk CSV, reload forecasting model
        bm.record_sale(items, filepath)
        
        # Recalculate inventory alerts
        alerts = bm.inventory_alerts(CURRENT_STOCK, forecast_days=7)
        summary = bm.get_summary_stats()
        
        return jsonify({
            "message": f"Successfully recorded transaction with {len(items)} items.",
            "stock": CURRENT_STOCK,
            "alerts": alerts,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API — sentiment analysis & feedback reviews
# ---------------------------------------------------------------------------
REVIEWS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.json")

def load_reviews():
    import json
    if not os.path.exists(REVIEWS_FILE) or os.path.getsize(REVIEWS_FILE) == 0:
        default_reviews = [
            {"product": "Coffee", "rating": 5, "comment": "Coffee is always fresh and hot. Best morning spot.", "sentiment": "positive", "flags": [], "date": "2026-07-03"},
            {"product": "Bread", "rating": 2, "comment": "The bread was really stale today, very dry.", "sentiment": "negative", "flags": ["stale", "dry"], "date": "2026-07-02"},
            {"product": "Cake", "rating": 3, "comment": "Nice place, but the cake was a bit too sweet.", "sentiment": "neutral", "flags": ["sweet"], "date": "2026-07-02"},
            {"product": "Pastry", "rating": 5, "comment": "Excellent pastry, loved the croissants!", "sentiment": "positive", "flags": [], "date": "2026-07-01"}
        ]
        with open(REVIEWS_FILE, 'w') as f:
            json.dump(default_reviews, f, indent=2)
        return default_reviews
        
    try:
        with open(REVIEWS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_review_item(review):
    import json
    reviews = load_reviews()
    reviews.insert(0, review) # insert at start
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f, indent=2)

def analyze_sentiment(text: str) -> tuple[str, list[str]]:
    text_lower = text.lower()
    pos_words = ["fresh", "delicious", "best", "excellent", "loved", "great", "good", "perfect", "tasty", "amazing", "yummy"]
    neg_words = ["stale", "dry", "sweet", "cold", "burnt", "bad", "poor", "slow", "hard", "expensive", "disappointed", "worst"]
    
    pos_score = sum(1 for w in pos_words if w in text_lower)
    neg_score = sum(1 for w in neg_words if w in text_lower)
    
    score = pos_score - neg_score
    if score > 0:
        sentiment = "positive"
    elif score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    flags = []
    for f in ["stale", "dry", "burnt", "cold", "hard", "slow"]:
        if f in text_lower:
            flags.append(f)
    if "sweet" in text_lower and "too sweet" in text_lower:
        flags.append("too sweet")
    elif "sweet" in text_lower:
        flags.append("sweet")
        
    return sentiment, flags

@app.route("/api/reviews", methods=["GET", "POST"])
def api_reviews():
    if request.method == "POST":
        data = request.get_json(force=True)
        product = data.get("product")
        rating = int(data.get("rating", 5))
        comment = data.get("comment", "").strip()
        
        if not product or not comment:
            return jsonify({"error": "Product and comment are required"}), 400
            
        sentiment, flags = analyze_sentiment(comment)
        
        review = {
            "product": product,
            "rating": rating,
            "comment": comment,
            "sentiment": sentiment,
            "flags": flags,
            "date": datetime.today().strftime("%Y-%m-%d")
        }
        
        save_review_item(review)
        
    # Get summary and review list
    reviews = load_reviews()
    
    # Compute product level alerts / metrics
    from collections import defaultdict
    prod_sentiment = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0, "total": 0, "flags": set()})
    
    for r in reviews:
        p = r["product"]
        s = r["sentiment"]
        prod_sentiment[p]["total"] += 1
        if s == "positive":
            prod_sentiment[p]["pos"] += 1
        elif s == "negative":
            prod_sentiment[p]["neg"] += 1
        else:
            prod_sentiment[p]["neu"] += 1
            
        for f in r.get("flags", []):
            prod_sentiment[p]["flags"].add(f)
            
    # Format prod_sentiment dict into standard JSON-serializable list
    alerts = []
    for p, stats in prod_sentiment.items():
        neg_pct = (stats["neg"] / stats["total"] * 100) if stats["total"] > 0 else 0
        if neg_pct > 25:
            alerts.append({
                "product": p,
                "negative_percentage": round(neg_pct, 1),
                "total_reviews": stats["total"],
                "key_issues": list(stats["flags"])
            })
            
    return jsonify({
        "reviews": reviews[:20],
        "alerts": alerts
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Bakery Management System")
    print("  Data: Bakery.csv (real transactions)")
    print("  Models: Linear Regression + Random Forest")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
