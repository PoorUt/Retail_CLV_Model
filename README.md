# 🛒 Retail CLV Intelligence Platform

A full-stack **Customer Lifetime Value (CLV) optimization platform** built on 2 years of UK retail transaction data. It combines probabilistic CLV models, machine learning churn prediction, and promotional uplift modelling into an interactive Streamlit dashboard.

---

## 📸 App Overview

The app has **7 interactive pages**:

| Page | Description |
|---|---|
| 📊 Overview | Portfolio KPIs, segment breakdown, CLV distribution, top customers |
| 🧩 Segmentation | Radar profiles, heatmaps, scatter matrix, box plots per segment |
| 💰 CLV Analysis | Pareto curve, CLV drivers, percentile bands, revenue mix |
| ⚠️ Churn Analysis | Risk matrix, churn distributions, recency analysis, at-risk table |
| 📐 RFM Analysis | 3D RFM scatter, correlation heatmap, purchase timing patterns |
| 🔎 Customer Explorer | Multi-filter drill-down, sortable table, CSV download |
| 🚀 Score a Customer | Real-time CLV, churn probability, uplift score & recommendations |

---

## 🧠 Models Used

| Model | Library | Purpose |
|---|---|---|
| **BG/NBD** | `lifetimes` | Predicts probability a customer is still alive & expected purchases |
| **Gamma-Gamma** | `lifetimes` | Predicts expected monetary value per transaction |
| **XGBoost Classifier** | `xgboost` | Churn probability (ROC-AUC: 0.9994) |
| **T-Learner** | `scikit-uplift` | Promotional uplift estimation |
| **K-Means (k=4)** | `scikit-learn` | Customer segmentation into 4 tiers |

---

## 👥 Customer Segments

| Segment | Customers | Criteria |
|---|---|---|
| 🟢 **Champions** | 144 (3.4%) | P(Alive) > 70% AND Total Spend > £500 |
| 🔵 **Loyal Customers** | 957 (22.9%) | P(Alive) > 50% |
| 🟠 **At Risk** | 2,198 (52.6%) | Churn Probability > 60% |
| 🔴 **Lost / Dormant** | 882 (21.1%) | All others |

---

## 📁 Project Structure

```
retail-clv-app/
│
├── app.py                  # Streamlit dashboard (7 pages)
├── utils.py                # Data cleaning, feature engineering, model utilities
├── retail_clv.ipynb        # Full ML pipeline notebook
├── requirements.txt        # Python dependencies
│
├── df_full.csv             # Processed customer dataset (4,181 customers × 20 features)
│
├── bgf.pkl                 # Trained BG/NBD model
├── ggf.pkl                 # Trained Gamma-Gamma model
├── churn_model.pkl         # Trained XGBoost churn classifier
├── uplift_model.pkl        # Trained T-Learner uplift model
├── scaler.pkl              # Fitted StandardScaler
├── kmeans.pkl              # Fitted K-Means model
│
├── templates/
│   └── index.html          # UI design reference (not served)
│
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-clv-app.git
cd retail-clv-app
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📊 Dataset

- **Source:** [UCI Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Period:** December 2009 – December 2011
- **Raw transactions:** 1,067,371 rows across 8 columns
- **After cleaning:** 802,932 transactions, 5,862 unique customers
- **Modelling subset:** 4,181 customers with ≥ 1 repeat purchase

> ⚠️ The raw `online_retail_II.xlsx` file is excluded from this repo (45 MB).  
> Run the notebook `retail_clv.ipynb` to regenerate `df_full.csv` and all `.pkl` files from scratch.

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| **App** | Streamlit, Plotly Express, Plotly Graph Objects |
| **CLV Models** | Lifetimes (BG/NBD, Gamma-Gamma) |
| **ML** | Scikit-learn, XGBoost, Scikit-uplift |
| **Data** | Pandas, NumPy |
| **Serialisation** | Dill (pickle) |
| **Notebook** | Jupyter, IPyKernel |

---

## 📈 Key Results

- **Mean 90-day CLV:** £104.31
- **Median 90-day CLV:** £41.99
- **Highest individual CLV:** £21,705
- **Churn model ROC-AUC:** 0.9994
- **Average P(Alive):** 81%
- **Overall churn rate:** 14.7%

---

## 🌐 Live Demo

> Deploy on **[Streamlit Cloud](https://share.streamlit.io)** for free:
> 1. Push this repo to GitHub
> 2. Go to share.streamlit.io → New App
> 3. Select repo → branch `main` → file `app.py`
> 4. Click **Deploy** ✅

---

## 📄 License

This project is for educational and portfolio purposes.  
Dataset © UCI Machine Learning Repository.
