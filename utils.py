# All reusable functions for the Retail CLV Optimization project.
# Both retail_clv.ipynb and app.py import from here.

import os
import numpy as np
import pandas as pd
import dill
 
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#===========================================
# 1. Data Cleaning
#===========================================

def clean_data(df):
    """
    this function cleans data. The following steps are performed
    - Standardise column names
    - Drop rows missing Customer ID
    - Remove cancellations (Invoice starts with 'C')
    - Remove negative / zero quantities and prices
    - Remove non-product stock codes
    - Add derived columns: total_price, date parts 

    """
    df = df.copy()

    # Standardise column names 
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Drop rows with no customer ID
    df.dropna(subset=['customer_id'], inplace=True)
    df['customer_id'] = df['customer_id'].astype(int).astype(str)

    # Remove cancellations
    df = df[~df['invoice'].astype(str).str.startswith('C')]

    # Keep only valid positive quantities and prices
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]

    # Remove non-product stock codes
    non_product = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT']
    df = df[~df['stockcode'].astype(str).str.upper().isin(non_product)]

    # Parse dates and derive useful columns
    df['invoicedate']   = pd.to_datetime(df['invoicedate'])
    df['total_price']   = df['quantity'] * df['price']
    df['invoice_year']  = df['invoicedate'].dt.year
    df['invoice_month'] = df['invoicedate'].dt.month
    df['invoice_dow']   = df['invoicedate'].dt.dayofweek
    df['invoice_hour']  = df['invoicedate'].dt.hour
 
    df.reset_index(drop=True, inplace=True)
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
"""
in this context we will use Python's "liftimes" Library.  lifetimes is a popular Python library used to analyze Customer Lifetime Value (CLV). 
It helps businesses predict how much a customer will spend in the future and how likely they are to "churn" (stop buying). 
It is based on the "Buy Til You Die" (BTYD) statistical models, which are specifically designed for non-contractual businesses 
(like retail or e-commerce) where customers can stop coming back at any time without telling you.

Core Concepts of the lifetimes Package
The library primarily looks at three pieces of data for every customer, often called RFM metrics:

    Recency: How long ago was their last purchase?
    Frequency: How many repeat purchases have they made?
    T (Age): How much time has passed since their very first purchase?

What can you do with it?

The package provides tools to answer specific business questions:

    Predict Future Transactions: "How many purchases will this customer make in the next 90 days?" (Using the BG/NBD model).
    Probability of Being Alive: "Based on their behavior, what is the probability that this customer is still a 'customer' and hasn't left us forever?"
    Customer Value: "What is the expected monetary value of this customer over the next year?" (Using the Gamma-Gamma model).

Visualizing the Results

The package includes built-in plotting functions that are very useful for presentations. For example, the Frequency/Recency Matrix shows you 
exactly which customers are your "Best" (bought frequently and recently) and which ones are "At Risk" (bought a lot in the past but haven't been back in a long time).

Why use it instead of regular Machine Learning?
Traditional Machine Learning (like Random Forest) often struggles with customer behavior because customer data is "sparse"—most people only buy once or twice. 
The statistical models in lifetimes are mathematically tuned to handle these long gaps in time much better than standard algorithms.

"""
def build_rfm_summary(df, snapshot_date):
    """
    Build the standard lifetimes RFM summary table.
    Filters to customers with at least 1 repeat purchase (BG/NBD requirement).
    """
    rfm = summary_data_from_transaction_data(
        df,
        customer_id_col='customer_id',
        datetime_col='invoicedate',
        monetary_value_col='total_price',
        observation_period_end=snapshot_date,
        freq='D',
    )
    rfm = rfm[rfm['frequency'] > 0].copy()
    return rfm

def build_extended_rfm(df, snapshot_date):
    """
    Build extended RFM features beyond basic recency / frequency / monetary.
    Adds basket behaviour, product diversity, temporal patterns, purchase velocity.
    """
    grp = df.groupby('customer_id')
 
    features = pd.DataFrame({
        'recency_days'       : (snapshot_date - grp['invoicedate'].max()).dt.days,
        'frequency'          : grp['invoice'].nunique(),
        'monetary_total'     : grp['total_price'].sum(),
        'monetary_avg'       : grp['total_price'].mean(),
        'avg_basket_size'    : grp['quantity'].mean(),
        'avg_items_per_order': grp.apply(lambda x: x.groupby('invoice')['quantity'].sum().mean()),
        'unique_products'    : grp['stockcode'].nunique(),
        'preferred_dow'      : grp['invoice_dow'].agg(lambda x: x.mode()[0]),
        'preferred_hour'     : grp['invoice_hour'].agg(lambda x: x.mode()[0]),
        'tenure_days'        : (grp['invoicedate'].max() - grp['invoicedate'].min()).dt.days,
        'purchase_velocity'  : grp.apply(
            lambda x: x.drop_duplicates('invoice')
                       .sort_values('invoicedate')['invoicedate']
                       .diff().dt.days.mean()
        ),
    })

    features['purchase_velocity'].fillna(features['tenure_days'], inplace=True)
    features.reset_index(inplace=True)
    return features

# =============================================================================
# 3. CLV MODELLING
# =============================================================================
 
def fit_bgnbd(rfm_summary):
    """
    Fit the BG/NBD model on the RFM summary table.
    Returns the fitted BetaGeoFitter object.
    """
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
        verbose=False,
    )
    return bgf
 
 
def fit_gamma_gamma(rfm_summary):
    """
    Fit the Gamma-Gamma spend model on the RFM summary table.
    Returns the fitted GammaGammaFitter object.
    """
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rfm_summary['frequency'], rfm_summary['monetary_value'])
    return ggf
 
 
def compute_clv(bgf, ggf, rfm_summary, forecast_days=90, forecast_months=3,
                discount_rate=0.01, profit_margin=0.20):
    """
    Compute 90-day CLV, predicted purchases, and P(alive) for each customer.
    Adds columns directly to rfm_summary and returns it.
    """
    rfm_summary = rfm_summary.copy()
 
    rfm_summary['predicted_purchases_90d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        forecast_days,
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
    )
 
    rfm_summary['prob_alive'] = bgf.conditional_probability_alive(
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
    )
 
    rfm_summary['clv_90d'] = ggf.customer_lifetime_value(
        bgf,
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
        rfm_summary['monetary_value'],
        time=forecast_months,
        discount_rate=discount_rate,
    ) * profit_margin
 
    return rfm_summary
 
 
# =============================================================================
# 4. SEGMENTATION
# =============================================================================

CLUSTER_FEATURES = [
    'recency_days', 'frequency', 'monetary_total',
    'monetary_avg', 'tenure_days', 'purchase_velocity',
    'unique_products', 'predicted_purchases_90d',
    'prob_alive', 'clv_90d',
]
 
def fit_segments(df_full, n_clusters=4, random_state=42):
    """
    Fit K-Means segmentation on the full feature table.
    Returns df_full with 'segment' and 'segment_label' columns added,
    plus the fitted scaler (needed to transform new customers at score time).
    """
    X = df_full[CLUSTER_FEATURES].fillna(0)
    
    # Cap extreme outliers at 99th percentile so they don't distort clusters
    for col in CLUSTER_FEATURES:
        cap = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=cap)


    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df_full = df_full.copy()
    df_full['segment'] = kmeans.fit_predict(X_scaled)
 
    # Auto-label segments by average CLV (highest CLV = Champions)
    seg_clv_rank = (
        df_full.groupby('segment')['clv_90d']
        .mean()
        .rank(ascending=False)
        .astype(int)
    )
    label_map = {
        seg: name for seg, name in zip(
            seg_clv_rank.sort_values().index,
            ['Champions', 'Loyal Customers', 'At Risk', 'Lost / Dormant']
        )
    }
    df_full['segment_label'] = df_full['segment'].map(label_map)
 
    return df_full, scaler, kmeans
 
 
def assign_segment_label(prob_alive, monetary_total, churn_prob):
    """
    Rule-based segment label used at API score time.
    Mirrors the cluster labels assigned during training.
    """
    if prob_alive > 0.7 and monetary_total > 500:
        return 'Champions'
    elif prob_alive > 0.5:
        return 'Loyal Customers'
    elif churn_prob > 0.6:
        return 'At Risk'
    else:
        return 'Lost / Dormant'
 
 
# =============================================================================
# 5. CHURN LABEL
# =============================================================================
 
def add_churn_label(df_full, churn_window=90):
    """
    Add a binary churn label.
    A customer is churned if they haven't purchased in churn_window days
    AND their BG/NBD prob_alive is below 0.5.
    """
    df_full = df_full.copy()
    df_full['churned'] = (
        (df_full['recency_days'] > churn_window) &
        (df_full['prob_alive']   < 0.5)
    ).astype(int)
    return df_full
 
 
# =============================================================================
# 6. BUSINESS SIMULATION
# =============================================================================
 
def simulate_roi(df_eval, sort_col, n_target,
                 promo_cost=2.0, conversion_rev=35.0,
                 ascending=False, label=''):
    """
    Simulate promotional ROI for a targeting strategy.
 
    df_eval must have columns: converted, treatment, and the sort_col.
    n_target = number of customers to target.
    """
    targeted  = df_eval.nsmallest(n_target, sort_col) if ascending else df_eval.nlargest(n_target, sort_col)
    treated   = targeted[targeted['treatment'] == 1]
    revenue   = treated['converted'].sum() * conversion_rev
    cost      = n_target * promo_cost
    roi       = (revenue - cost) / cost * 100
    print(f'[{label:<28}]  Revenue: £{revenue:>7,.0f}  |  Cost: £{cost:>6,.0f}  |  ROI: {roi:.1f}%')
    return {'revenue': revenue, 'cost': cost, 'roi': roi}
 
 
# =============================================================================
# 7. MODEL PERSISTENCE
# =============================================================================
 
def save_models(models: dict, path='.'):
    for name, model in models.items():
        filepath = os.path.join(path, f'{name}.pkl')
        with open(filepath, 'wb') as f:
            dill.dump(model, f)
        print(f'Saved: {filepath}')
 
 
def load_models(names: list, path='.'):
    models = {}
    for name in names:
        filepath = os.path.join(path, f'{name}.pkl')
        with open(filepath, 'rb') as f:
            models[name] = dill.load(f)
    return models