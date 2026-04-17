import os
import io
import base64
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─── Load & Prepare Data (runs once at startup) ─────────────────────────────

df_raw = pd.read_csv("personal_spending_dataset.csv")

def preprocess_data(df):
    """Clean and encode the dataset, engineer new features."""
    data = df.copy()

    # Step 1: Map binary Yes/No columns to 1/0
    for col in ["investment", "emergency_fund"]:
        data[col] = data[col].map({"Yes": 1, "No": 0})

    # Step 2: Label-encode categorical columns
    cat_cols = ["gender", "occupation", "city", "income_source",
                "credit_card_usage", "financial_stress"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le

    # Step 3: Drop rows with missing target
    data = data.dropna(subset=["financial_health_score"])

    # Step 4: Cap debt outliers at 99th percentile
    cap = data["debt"].quantile(0.99)
    data["debt"] = data["debt"].clip(upper=cap)

    # Step 5: Feature engineering
    expense_cols = ["housing_expense", "food_expense", "transport_expense",
                    "entertainment_expense", "shopping_expense", "healthcare_expense"]
    data["total_expenses"] = data[expense_cols].sum(axis=1)
    data["expense_to_income_ratio"] = data["total_expenses"] / (data["monthly_income"] + 1)

    return data, le_dict


def train_all_models(df):
    """Train 4 regression models and return results."""
    data, le_dict = preprocess_data(df)

    feature_cols = [
        "age", "gender", "occupation", "city", "monthly_income",
        "income_source", "savings_rate", "debt", "housing_expense",
        "food_expense", "transport_expense", "entertainment_expense",
        "shopping_expense", "healthcare_expense", "credit_card_usage",
        "investment", "emergency_fund", "financial_stress",
        "total_expenses", "expense_to_income_ratio"
    ]

    X = data[feature_cols]
    y = data["financial_health_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models_def = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42), False),
    }

    results = {}
    trained = {}

    for name, (model, use_scaled) in models_def.items():
        if use_scaled:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        results[name] = {
            "RMSE": round(rmse, 4),
            "MAE":  round(mae, 4),
            "R2":   round(r2, 4),
            "y_pred": y_pred.tolist(),
            "y_test": y_test.tolist(),
        }
        trained[name] = (model, use_scaled)

    return trained, results, scaler, feature_cols, le_dict, data


# Train once at startup
TRAINED_MODELS, RESULTS, SCALER, FEATURE_COLS, LE_DICT, PROC_DATA = train_all_models(df_raw)
BEST_MODEL = max(RESULTS, key=lambda m: RESULTS[m]["R2"])


# ─── Helper: convert matplotlib figure to base64 PNG ────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    stats = {
        "rows": len(df_raw),
        "cols": len(df_raw.columns),
        "models": len(RESULTS),
        "best_model": BEST_MODEL,
        "best_r2": RESULTS[BEST_MODEL]["R2"],
    }
    return render_template("index.html", stats=stats)


@app.route("/data")
def data_overview():
    sample = df_raw.head(8).to_html(
        classes="data-table", border=0, index=False
    )
    desc = df_raw.describe().round(2).to_html(
        classes="data-table", border=0
    )
    missing = df_raw.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing": missing.values,
        "Pct": (missing.values / len(df_raw) * 100).round(2)
    })
    missing_html = missing_df[missing_df["Missing"] > 0].to_html(
        classes="data-table", border=0, index=False
    ) if missing.sum() > 0 else "<p class='ok-msg'>✅ No missing values found.</p>"

    # Target distribution chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df_raw["financial_health_score"], bins=40,
            color="#3b82f6", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Financial Health Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Target Variable", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#f8fafc")
    dist_chart = fig_to_b64(fig)

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    numeric = df_raw.select_dtypes(include=np.number)
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, ax=ax2, annot_kws={"size": 7})
    ax2.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_chart = fig_to_b64(fig2)

    return render_template("data.html",
        sample=sample, desc=desc, missing_html=missing_html,
        dist_chart=dist_chart, corr_chart=corr_chart,
        shape=(df_raw.shape[0], df_raw.shape[1])
    )


@app.route("/preprocessing")
def preprocessing():
    # Before/after debt boxplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].boxplot(df_raw["debt"].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#fca5a5", color="#ef4444"),
                    medianprops=dict(color="white", linewidth=2))
    axes[0].set_title("Debt — Before Capping", fontsize=11, fontweight="bold")
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].boxplot(PROC_DATA["debt"], vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#6ee7b7", color="#10b981"),
                    medianprops=dict(color="white", linewidth=2))
    axes[1].set_title("Debt — After Capping (99th pct)", fontsize=11, fontweight="bold")
    axes[1].spines[["top","right"]].set_visible(False)

    for ax in axes:
        ax.set_facecolor("#f8fafc")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    boxplot_img = fig_to_b64(fig)

    # New features preview
    feat_preview = PROC_DATA[["monthly_income", "total_expenses", "expense_to_income_ratio"]].head(6)
    feat_html = feat_preview.round(2).to_html(classes="data-table", border=0, index=False)

    feature_list = FEATURE_COLS

    return render_template("preprocessing.html",
        boxplot_img=boxplot_img,
        feat_html=feat_html,
        feature_list=feature_list,
        n_train=int(len(PROC_DATA) * 0.8),
        n_test=int(len(PROC_DATA) * 0.2),
        debt_cap=round(df_raw["debt"].quantile(0.99), 2)
    )


@app.route("/training")
def training():
    model_info = {
        "Linear Regression": {
            "icon": "📐",
            "desc": "Baseline model that fits a straight-line relationship between features and the target score.",
            "params": "Default — no hyperparameters",
            "pros": "Fast, easy to interpret, good baseline",
            "cons": "Assumes linear relationships — will underfit complex patterns",
        },
        "Ridge Regression": {
            "icon": "🔗",
            "desc": "Linear regression with L2 regularization to reduce overfitting caused by correlated features.",
            "params": "alpha = 1.0 (regularization strength)",
            "pros": "Handles multicollinearity, slightly better than plain linear",
            "cons": "Still limited by linearity assumption",
        },
        "Random Forest": {
            "icon": "🌲",
            "desc": "Ensemble of 100 decision trees using bagging — each tree learns from a random subset of data.",
            "params": "n_estimators=100, max_depth=10, random_state=42",
            "pros": "High accuracy, handles non-linearity, gives feature importance",
            "cons": "Slower training, less interpretable than linear models",
        },
        "Gradient Boosting": {
            "icon": "🚀",
            "desc": "Sequentially builds trees that each correct errors of the previous one.",
            "params": "n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42",
            "pros": "Often best performance, flexible and powerful",
            "cons": "Slowest to train, most hyperparameters to tune",
        },
    }

    summary = [
        {"model": m, "rmse": RESULTS[m]["RMSE"], "mae": RESULTS[m]["MAE"], "r2": RESULTS[m]["R2"]}
        for m in RESULTS
    ]

    return render_template("training.html",
        model_info=model_info,
        summary=summary,
        best_model=BEST_MODEL
    )


@app.route("/evaluation")
def evaluation():
    model_name = request.args.get("model", BEST_MODEL)
    res = RESULTS[model_name]
    y_test = np.array(res["y_test"])
    y_pred = np.array(res["y_pred"])

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=12, color="#3b82f6")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
    ax.set_xlabel("Actual Score", fontsize=11)
    ax.set_ylabel("Predicted Score", fontsize=11)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_alpha(0)
    scatter_img = fig_to_b64(fig)

    # Residuals
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(residuals, bins=40, color="#8b5cf6", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="#ef4444", linestyle="--", lw=2)
    ax2.set_xlabel("Residual (Actual − Predicted)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_facecolor("#f8fafc")
    fig2.patch.set_alpha(0)
    resid_img = fig_to_b64(fig2)

    # Feature importance
    rf_model, _ = TRAINED_MODELS["Random Forest"]
    importance = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importance})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(feat_df)))
    ax3.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
    ax3.set_xlabel("Importance", fontsize=11)
    ax3.set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.set_facecolor("#f8fafc")
    fig3.patch.set_alpha(0)
    plt.tight_layout()
    feat_img = fig_to_b64(fig3)

    # Model comparison bar chart
    model_names = list(RESULTS.keys())
    rmse_vals = [RESULTS[m]["RMSE"] for m in model_names]
    mae_vals  = [RESULTS[m]["MAE"]  for m in model_names]
    r2_vals   = [RESULTS[m]["R2"]   for m in model_names]

    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))
    short = [m.replace(" ", "\n") for m in model_names]
    for ax, vals, label, color in zip(
        axes4,
        [rmse_vals, mae_vals, r2_vals],
        ["RMSE ↓", "MAE ↓", "R² ↑"],
        ["#ef4444", "#f97316", "#22c55e"]
    ):
        ax.bar(short, vals, color=color, edgecolor="white", width=0.5)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig4.patch.set_alpha(0)
    plt.tight_layout()
    compare_img = fig_to_b64(fig4)

    return render_template("evaluation.html",
        model_name=model_name,
        all_models=list(RESULTS.keys()),
        rmse=res["RMSE"], mae=res["MAE"], r2=res["R2"],
        scatter_img=scatter_img, resid_img=resid_img,
        feat_img=feat_img, compare_img=compare_img
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            model_name = form_data.get("model", BEST_MODEL)

            def safe_enc(le, val):
                classes = list(le.classes_)
                if val not in classes:
                    val = classes[0]
                return int(le.transform([val])[0])

            age            = float(form_data["age"])
            gender         = safe_enc(LE_DICT["gender"],          form_data["gender"])
            occupation     = safe_enc(LE_DICT["occupation"],      form_data["occupation"])
            city           = safe_enc(LE_DICT["city"],            form_data["city"])
            monthly_income = float(form_data["monthly_income"])
            income_source  = safe_enc(LE_DICT["income_source"],   form_data["income_source"])
            savings_rate   = float(form_data["savings_rate"])
            debt           = float(form_data["debt"])
            housing        = float(form_data["housing_expense"])
            food           = float(form_data["food_expense"])
            transport      = float(form_data["transport_expense"])
            entertainment  = float(form_data["entertainment_expense"])
            shopping       = float(form_data["shopping_expense"])
            healthcare     = float(form_data["healthcare_expense"])
            credit_card    = safe_enc(LE_DICT["credit_card_usage"], form_data["credit_card_usage"])
            investment     = 1 if form_data.get("investment") == "Yes" else 0
            emergency_fund = 1 if form_data.get("emergency_fund") == "Yes" else 0
            stress         = safe_enc(LE_DICT["financial_stress"], form_data["financial_stress"])

            total_expenses = housing + food + transport + entertainment + shopping + healthcare
            exp_ratio = total_expenses / (monthly_income + 1)

            x = np.array([[
                age, gender, occupation, city, monthly_income, income_source,
                savings_rate, debt, housing, food, transport, entertainment,
                shopping, healthcare, credit_card, investment, emergency_fund,
                stress, total_expenses, exp_ratio
            ]])

            model, use_scaled = TRAINED_MODELS[model_name]
            if use_scaled:
                x = SCALER.transform(x)

            score = float(model.predict(x)[0])
            score = max(0.0, score)

            if score >= 30:
                level = "good"
                label = "Good Financial Health 🟢"
                tip = "You're managing your finances well. Keep saving and investing!"
            elif score >= 15:
                level = "moderate"
                label = "Moderate Financial Health 🟡"
                tip = "There's room to improve. Consider reviewing your expense-to-income ratio."
            else:
                level = "low"
                label = "Low Financial Health 🔴"
                tip = "Consider reducing debt, cutting discretionary expenses, and building an emergency fund."

            prediction = {
                "score": round(score, 2),
                "level": level,
                "label": label,
                "tip": tip,
                "model": model_name,
                "total_expenses": round(total_expenses, 2),
                "exp_ratio": round(exp_ratio, 3),
            }

        except Exception as e:
            error = str(e)

    return render_template("predict.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        models=list(RESULTS.keys()),
        best_model=BEST_MODEL
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
