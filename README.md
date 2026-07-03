# Smart Bakery Management System with ML

An AI-powered dashboard designed to help bakery owners predict product demand, optimize inventory stock levels, minimize daily wastage, analyze customer behavior, and review product feedback.

---

## Complete Setup Guide (From Scratch)

This guide is designed for beginners. Follow these steps if you have **no Python or Git installed** on your laptop.

### Prerequisites

#### Step 1: Install Git
Git is used to download (clone) the project code.
1. Download Git from [git-scm.com/downloads](https://git-scm.com/downloads) (Select the **Windows** version).
2. Open the downloaded installer and click **Next** through all steps. Keep the default settings.
3. Once finished, open your search bar, type **Command Prompt** (cmd), and press Enter.
4. Type `git --version` and press Enter to verify it is installed.

#### Step 2: Install Python (Crucial Step)
Python runs the backend server and machine learning models.
1. Download Python 3.12 (or latest version) from [python.org/downloads](https://www.python.org/downloads/) (Select the **Windows** installer).
2. Open the installer file.
3. **IMPORTANT:** At the bottom of the installation window, check the box that says **"Add python.exe to PATH"**. If you skip this, Windows won't recognize Python commands.
4. Click **Install Now**.
5. Once complete, close and reopen your **Command Prompt** (cmd).
6. Type `python --version` and press Enter to verify the installation.

---

### Project Installation

#### Step 3: Download the Project
1. Open **Command Prompt** (cmd).
2. Navigate to your Desktop by typing:
   ```bash
   cd Desktop
   ```
3. Clone this repository onto your laptop:
   ```bash
   git clone https://github.com/efx-suresh7/smart_bakery_ml.git
   ```
4. Enter the project folder:
   ```bash
   cd smart_bakery_ml
   ```

#### Step 4: Create a Virtual Environment
A virtual environment keeps the project dependencies isolated from the rest of your laptop.
1. Inside the project folder in Command Prompt, run:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - **On Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **On macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```
   *(You should now see `(.venv)` displayed at the beginning of your command line prompt).*

#### Step 5: Install Required Packages
Install the ML models and web server libraries:
1. Run this command:
   ```bash
   pip install -r requirements.txt
   ```
   *(This downloads and installs Flask, pandas, scikit-learn, numpy, and other utilities).*

---

### Running the Application

#### Step 6: Start the Server
1. Inside your active environment, run:
   ```bash
   python app.py
   ```
2. You will see output showing that the server has started successfully:
   ```text
   =======================================================
     Smart Bakery Management System
     Data: Bakery.csv (real transactions)
     Models: Linear Regression + Random Forest
     URL: http://127.0.0.1:5000
   =======================================================
    * Running on http://127.0.0.1:5000
   ```

#### Step 7: Open the Dashboard
1. Open your web browser (Chrome, Edge, Firefox, etc.).
2. Go to the address: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
3. Enjoy exploring the dashboard!

---

## Dashboard Overview & Features

Once the dashboard is open, you will have access to:
- **Dashboard Overview:** Displays overall transaction counters, average items sold per day, and historical trend lines.
- **POS (Sales Entry):** Enter items sold in real-time. Submitting a sale appends it directly to the active CSV file and automatically retrains the machine learning models.
- **Predictions & Forecasting:** Interactive forecasts showing predicted daily sales for the next 7, 14, or 30 days. Includes a date picker to predict demand for any specific date.
- **Model Comparison:** Comparison metrics (R² accuracy, Mean Absolute Error) between the Linear Regression and Random Forest models.
- **Customer Insights:** Visual representation of customer groups using **K-Means Clustering** and frequently purchased combos mined via **Apriori-style analysis**.
- **Inventory & Wastage:** Evaluates current stock against expected end-of-day demand and alerts you of potential wastage, recommending discount percentages (e.g., "30% off at 16:00 PM") to clear inventory.
- **Feedback & Sentiment:** Analyze reviews left by customers. System flags products with high negative feedback and displays extracted issue words (like "stale", "dry", or "burnt").

---

## File Structure

```text
smart_bakery_ml/
├── Bakery.csv            # The historical transactions dataset
├── app.py                # Flask backend server containing all API routes
├── data_loader.py        # Preprocess transactions and aggregates dates
├── ml_models.py          # Random Forest, Linear Regression, K-Means & Apriori engines
├── requirements.txt      # List of dependencies
├── static/               # CSS styles and JavaScript charts
└── templates/            # HTML frontend layouts
```
