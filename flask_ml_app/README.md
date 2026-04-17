# 💰 Financial Health Score Predictor — Flask ML Web App
## Mid-Term Project | ML Web App Development

---

## Project Overview
This Flask web app predicts a person's **Financial Health Score (0–60+)**
using regression models trained on a personal spending dataset.

**Algorithm: Regression** — the target is a continuous number, not a category.

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## Project Structure
```
flask_ml_app/
│
├── app.py                        ← Main Flask application
├── personal_spending_dataset.csv ← Dataset
├── requirements.txt
├── README.md
│
└── templates/
    ├── base.html         ← Sidebar layout & shared CSS
    ├── index.html        ← Home page
    ├── data.html         ← Data overview
    ├── preprocessing.html← Preprocessing steps
    ├── training.html     ← Model training
    ├── evaluation.html   ← Metrics & charts
    └── predict.html      ← Interactive prediction form
```

---

## Pages
| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Project overview, stats, workflow |
| `/data` | Data Overview | Sample, stats, charts, heatmap |
| `/preprocessing` | Preprocessing | All 6 preprocessing steps |
| `/training` | Model Training | 4 models with hyperparameters |
| `/evaluation` | Evaluation | RMSE/MAE/R², scatter, residuals |
| `/predict` | Predict | Form to predict your own score |

---

## Models Used
- **Linear Regression** — default params
- **Ridge Regression** — alpha=1.0
- **Random Forest** — n_estimators=100, max_depth=10
- **Gradient Boosting** — n_estimators=100, learning_rate=0.1

---

## Grading Criteria Coverage
| Criterion | Implementation |
|-----------|---------------|
| Dataset usefulness | Explained on Home page |
| Preprocessing | 6 steps on Preprocessing page |
| Training & testing | 4 models, hyperparameters shown |
| Evaluation metrics | RMSE, MAE, R² with charts |
| GUI | Sidebar layout, 6 pages, styled cards |
| Predict | Interactive form with color-coded result |
