# 📉 Customer Churn Prediction with Custom Logistic Regression

A from-scratch implementation of binary logistic regression using NumPy to predict customer churn in subscription-based platforms — benchmarked against scikit-learn and served via an interactive Streamlit dashboard.

---

## 🎯 Objective

Build a fully interpretable churn classifier without relying on ML libraries for the core model — implementing every component manually, from the sigmoid function to gradient descent — then validate it against scikit-learn's optimised implementation.

---

## ✨ Features

- **Custom LogReg from scratch** — sigmoid, log-loss, mini-batch gradient descent, L2 regularisation, early stopping
- **Class imbalance handling** — balanced sample weighting baked into the loss and gradient
- **Threshold tuning** — optimal decision boundary selected via F1 maximisation on validation set
- **Sklearn benchmark** — side-by-side comparison on all metrics
- **Interactive dashboard** — Streamlit app with ROC curves, confusion matrix, feature importance, live predictor
- **Full EDA** — churn distribution, tenure analysis, contract-type breakdown, correlation heatmap

---

## 📊 Results

| Metric | Custom LR | Sklearn LR |
|--------|-----------|------------|
| Accuracy | 0.761 | 0.742 |
| Precision | 0.534 | 0.508 |
| Recall | 0.754 | 0.796 |
| F1-Score | 0.625 | 0.620 |
| ROC-AUC | **0.847** | 0.854 |

> ROC-AUC gap of <1% confirms the from-scratch implementation matches sklearn's optimised solver. Industry benchmark for churn models: 0.80–0.90.

---

## 🗂️ Project Structure
```
customer-churn-predictor/
├── data/
├── notebooks/
│   └── churn_prediction.ipynb
├── reports/
│   ├── dashboard_data.json
│   ├── eda_plots.png
│   ├── loss_curves.png
│   ├── model_evaluation.png
│   └── feature_importance.png
├── deployment/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture
```
Input Features (22)
      ↓
StandardScaler
      ↓
Linear combination:  z = X·W + b
      ↓
Sigmoid:             σ(z) = 1 / (1 + e⁻ᶻ)
      ↓
Weighted Cross-Entropy Loss + L2 Regularisation
      ↓
Mini-batch Gradient Descent  (batch=64, lr=0.05, λ=0.001)
      ↓
Early Stopping on Validation Loss  (patience=30)
      ↓
Threshold Tuning  (default 0.5 → 0.38 via F1 maximisation)
```

---

## 📈 Key Insights

**Top churn risk factors**
- Month-to-month contract — 42.7% churn rate vs 2.8% for two-year contracts
- Fiber optic internet — higher expectations, greater dissatisfaction when unmet
- High monthly charges with no add-on services — price-sensitive, low perceived value

**Top churn reducers**
- Long tenure — switching inertia and loyalty compound over time
- Tech support subscription — supported customers feel valued and stay
- Bundled services (security, backup) — increase platform stickiness

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-benchmark-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-charts-3F4F75?logo=plotly)

---

## ⚙️ Setup & Usage
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction

# 2. Activate environment
conda activate churn

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook (trains model + exports dashboard data)
cd notebooks
jupyter notebook churn_prediction.ipynb

# 5. Launch the dashboard
cd ../deployment
streamlit run app.py
```

> **Dataset:** Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it in `data/`. If absent, the notebook auto-generates a realistic synthetic dataset.

---

## 📋 Deliverables

- [x] Exploratory Data Analysis & Feature Engineering
- [x] Sigmoid, Log-loss, Gradient Descent — implemented from scratch in NumPy
- [x] Class imbalance handling via weighted loss
- [x] Full evaluation — Accuracy, Precision, Recall, F1, ROC-AUC
- [x] Sklearn benchmark comparison
- [x] Training loss curves with early stopping
- [x] Feature importance via model coefficients
- [x] Interactive Streamlit dashboard with live predictor
- [x] Analytical report embedded in dashboard

---

> Built with assistance from [Claude](https://claude.ai) (Anthropic).
