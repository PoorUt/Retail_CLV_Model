# =============================================================================
# app.py  –  Retail CLV Intelligence Platform
# =============================================================================
# Run with:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_models, assign_segment_label
import io

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail CLV Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette (consistent across all charts) ────────────────────────────
SEG_COLORS = {
    "Champions":       "#10B981",
    "Loyal Customers": "#3B82F6",
    "At Risk":         "#F59E0B",
    "Lost / Dormant":  "#EF4444",
}

# ── Global Plotly theme ───────────────────────────────────────────────────────
def theme(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1e293b"), x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#475569", size=12),
        margin=dict(t=50 if title else 20, b=30, l=10, r=10),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
                     linecolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
                     linecolor="#e2e8f0", zeroline=False)
    return fig

# ── CSS – white dotted professional background ────────────────────────────────
st.markdown("""
<style>
/* ── Global: white dotted background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    background-image: radial-gradient(circle, #d1d5db 1.2px, transparent 1.2px);
    background-size: 22px 22px;
}
.stApp {
    background-color: transparent;
}

/* ── Main content pane ── */
.main .block-container {
    background: rgba(255,255,255,0.93);
    border-radius: 18px;
    padding: 2rem 2.8rem 3rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.07);
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div:not([data-testid]),
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] li { color: #cbd5e1; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9; }
[data-testid="stSidebar"] hr { border-color: #1e293b; }

/* Nav buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: #94a3b8 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.85rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 2px;
    transition: background 0.15s, color 0.15s;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
}
/* Active nav button */
[data-testid="stSidebar"] .stButton > button[data-nav-active="true"],
[data-testid="stSidebar"] .nav-active > button {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f8fafc;
    border-radius: 12px;
    padding: 4px 6px;
    gap: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #64748b;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1e293b !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: transform 0.1s, box-shadow 0.1s;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}

/* ── Headings ── */
h1 { color: #0f172a; font-weight: 800; letter-spacing: -0.02em; }
h2 { color: #1e293b; font-weight: 700; }
h3 { color: #334155; font-weight: 600; }

/* ── Section card helper ── */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Recommendation box ── */
.reco-box {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f8fafc; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA & MODELS – cached
# =============================================================================

@st.cache_resource
def load_all_models():
    return load_models(["bgf", "ggf", "churn_model", "uplift_model", "scaler", "kmeans"])


@st.cache_data
def load_data():
    df = pd.read_csv("df_full.csv")
    # Normalise segment label spacing
    if "segment_label" in df.columns:
        df["segment_label"] = df["segment_label"].str.strip()
    return df


models = load_all_models()
bgf          = models["bgf"]
ggf          = models["ggf"]
churn_model  = models["churn_model"]
uplift_model = models["uplift_model"]

try:
    df_full = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# =============================================================================
# SIDEBAR
# =============================================================================

# ── Session state for navigation ─────────────────────────────────────────────
PAGES = [
    "📊 Overview",
    "🧩 Segmentation",
    "💰 CLV Analysis",
    "⚠️ Churn Analysis",
    "📐 RFM Analysis",
    "🔎 Customer Explorer",
    "🚀 Score a Customer",
]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

with st.sidebar:
    st.markdown("### 🛒 CLV Intelligence")
    st.markdown("*Retail Customer Analytics*")
    st.markdown("---")

    for _p in PAGES:
        _is_active = st.session_state.page == _p
        _label = f"**{_p}**" if _is_active else _p
        if st.button(_label, key=f"nav__{_p}", use_container_width=True):
            st.session_state.page = _p
            st.rerun()

    st.markdown("---")
    st.markdown("**Models active**")
    st.markdown("- BG/NBD · Gamma-Gamma")
    st.markdown("- XGBoost Churn")
    st.markdown("- T-Learner Uplift")
    st.markdown("- K-Means (4 segments)")
    st.markdown("---")
    if data_loaded:
        n = len(df_full)
        st.markdown(f"**{n:,}** customers loaded")
    st.markdown("**Stack:** Python · Lifetimes · Streamlit")

page = st.session_state.page


# =============================================================================
# HELPER – no-data guard
# =============================================================================

def require_data():
    if not data_loaded:
        st.error("❌ `df_full.csv` not found. Run the notebook to generate it first.")
        st.stop()


# =============================================================================
# PAGE 1 – OVERVIEW DASHBOARD
# =============================================================================

if page == "📊 Overview":
    require_data()
    st.title("📊 Retail CLV Dashboard")
    st.markdown("High-level snapshot of customer value, behaviour, and risk across the entire base.")
    st.markdown("---")

    # ── KPI row ──────────────────────────────────────────────────────────────
    total_cust  = df_full["customer_id"].nunique() if "customer_id" in df_full.columns else len(df_full)
    total_clv   = df_full["clv_90d"].sum()
    avg_clv     = df_full["clv_90d"].mean()
    median_clv  = df_full["clv_90d"].median()
    churn_rate  = df_full["churned"].mean() * 100 if "churned" in df_full.columns else 0
    avg_palive  = df_full["prob_alive"].mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers",     f"{total_cust:,}")
    c2.metric("Portfolio 90d CLV",   f"£{total_clv:,.0f}")
    c3.metric("Mean CLV / Customer", f"£{avg_clv:,.2f}")
    c4.metric("Median CLV",          f"£{median_clv:,.2f}")
    c5.metric("Overall Churn Rate",  f"{churn_rate:.1f}%")
    c6.metric("Avg P(Alive)",        f"{avg_palive:.1f}%")

    st.markdown("---")

    # ── Row 1: Segment pie + CLV histogram ───────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Customer Segments</p>', unsafe_allow_html=True)
        seg_counts = df_full["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(seg_counts, names="Segment", values="Count",
                     color="Segment", color_discrete_map=SEG_COLORS, hole=0.45)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          hovertemplate="<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>")
        theme(fig, height=370)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">90-Day CLV Distribution</p>', unsafe_allow_html=True)
        cap = df_full["clv_90d"].quantile(0.99)
        fig = px.histogram(df_full[df_full["clv_90d"] < cap], x="clv_90d",
                           color="segment_label", nbins=60,
                           color_discrete_map=SEG_COLORS, barmode="overlay",
                           labels={"clv_90d": "CLV (£)", "segment_label": "Segment"})
        fig.update_traces(opacity=0.75)
        theme(fig, height=370)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Avg CLV + Churn by segment ────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-header">Average CLV by Segment</p>', unsafe_allow_html=True)
        agg = (df_full.groupby("segment_label")["clv_90d"].mean()
               .reset_index().sort_values("clv_90d"))
        agg.columns = ["Segment", "Avg CLV (£)"]
        fig = px.bar(agg, x="Avg CLV (£)", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=SEG_COLORS,
                     text="Avg CLV (£)")
        fig.update_traces(texttemplate="£%{text:.0f}", textposition="outside")
        fig.update_layout(showlegend=False)
        theme(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-header">Churn Rate by Segment</p>', unsafe_allow_html=True)
        if "churned" in df_full.columns:
            churn_seg = (df_full.groupby("segment_label")["churned"].mean()
                         .mul(100).reset_index().sort_values("churned"))
            churn_seg.columns = ["Segment", "Churn Rate (%)"]
            fig = px.bar(churn_seg, x="Churn Rate (%)", y="Segment", orientation="h",
                         color="Segment", color_discrete_map=SEG_COLORS,
                         text="Churn Rate (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False)
            theme(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Scatter CLV vs P(Alive) + Top 10 customers ────────────────────
    col_e, col_f = st.columns([3, 2])

    with col_e:
        st.markdown('<p class="section-header">CLV vs. Probability Alive</p>', unsafe_allow_html=True)
        cap95 = df_full["clv_90d"].quantile(0.95)
        plot_df = df_full[df_full["clv_90d"] < cap95].copy()
        fig = px.scatter(plot_df, x="prob_alive", y="clv_90d",
                         color="segment_label", color_discrete_map=SEG_COLORS,
                         opacity=0.6, size_max=8,
                         labels={"prob_alive": "P(Alive)", "clv_90d": "CLV (£)",
                                 "segment_label": "Segment"},
                         hover_data=["frequency", "recency_days"])
        theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown('<p class="section-header">Top 10 Customers by CLV</p>', unsafe_allow_html=True)
        top10_cols = ["customer_id", "clv_90d", "segment_label", "prob_alive"] \
            if "customer_id" in df_full.columns else ["clv_90d", "segment_label", "prob_alive"]
        top10 = df_full.nlargest(10, "clv_90d")[top10_cols].reset_index(drop=True)
        top10.index += 1
        st.dataframe(
            top10.style.format({
                "clv_90d":    "£{:.2f}",
                "prob_alive": "{:.1%}",
            }),
            use_container_width=True,
            height=345,
        )

    st.markdown("---")

    # ── Row 4: Predicted purchases violin + P(Alive) histogram ───────────────
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<p class="section-header">Predicted Purchases (90d) by Segment</p>',
                    unsafe_allow_html=True)
        fig = px.violin(df_full, x="segment_label", y="predicted_purchases_90d",
                        color="segment_label", color_discrete_map=SEG_COLORS,
                        box=True, points=False,
                        labels={"predicted_purchases_90d": "Predicted Purchases",
                                "segment_label": "Segment"})
        fig.update_layout(showlegend=False)
        theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_h:
        st.markdown('<p class="section-header">P(Alive) Distribution by Segment</p>',
                    unsafe_allow_html=True)
        fig = px.box(df_full, x="segment_label", y="prob_alive",
                     color="segment_label", color_discrete_map=SEG_COLORS,
                     points=False,
                     labels={"prob_alive": "P(Alive)", "segment_label": "Segment"})
        fig.update_layout(showlegend=False)
        theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2 – SEGMENTATION
# =============================================================================

elif page == "🧩 Segmentation":
    require_data()
    st.title("🧩 Customer Segmentation Analysis")
    st.markdown("Deep-dive into how your 4 customer tiers compare across every behavioural dimension.")
    st.markdown("---")

    seg_summary = (
        df_full.groupby("segment_label")
        .agg(
            Customers=("clv_90d", "count"),
            Avg_CLV=("clv_90d", "mean"),
            Median_CLV=("clv_90d", "median"),
            Total_CLV=("clv_90d", "sum"),
            Avg_PAlive=("prob_alive", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Recency=("recency_days", "mean"),
            Avg_Spend=("monetary_total", "mean"),
            Avg_Velocity=("purchase_velocity", "mean"),
        )
        .reset_index()
    )

    # ── Summary metrics per segment ───────────────────────────────────────────
    cols = st.columns(4)
    for i, seg in enumerate(["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"]):
        row = seg_summary[seg_summary["segment_label"] == seg]
        if row.empty:
            continue
        row = row.iloc[0]
        color_map = {"Champions": "#10B981", "Loyal Customers": "#3B82F6",
                     "At Risk": "#F59E0B", "Lost / Dormant": "#EF4444"}
        c = color_map[seg]
        cols[i].markdown(f"""
        <div style="background:white;border-radius:14px;padding:1.1rem 1.2rem;
                    border-top:4px solid {c};box-shadow:0 2px 12px rgba(0,0,0,0.06);">
          <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;
                      color:#64748b;font-weight:600;">{seg}</div>
          <div style="font-size:1.7rem;font-weight:800;color:#1e293b;line-height:1.2;">
            {int(row.Customers):,}</div>
          <div style="font-size:0.78rem;color:#64748b;margin-top:.2rem;">customers</div>
          <hr style="border-color:#f1f5f9;margin:.6rem 0;">
          <div style="font-size:0.82rem;color:#334155;">
            Avg CLV: <b>£{row.Avg_CLV:.0f}</b><br>
            P(Alive): <b>{row.Avg_PAlive:.1%}</b><br>
            Avg Spend: <b>£{row.Avg_Spend:.0f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distributions", "🕷️ Radar Profiles", "🌡️ Heatmap", "📈 Scatter Matrix"])

    with tab1:
        metrics = {
            "clv_90d":              "90d CLV (£)",
            "monetary_total":       "Total Spend (£)",
            "frequency":            "Purchase Frequency",
            "recency_days":         "Recency (days)",
            "purchase_velocity":    "Avg Days Between Orders",
            "unique_products":      "Unique Products",
            "tenure_days":          "Tenure (days)",
            "predicted_purchases_90d": "Predicted Purchases (90d)",
        }
        sel = st.selectbox("Metric to compare", list(metrics.keys()),
                           format_func=lambda k: metrics[k])
        cap99 = df_full[sel].quantile(0.99)
        plot_df = df_full[df_full[sel] < cap99]
        fig = px.box(plot_df, x="segment_label", y=sel, color="segment_label",
                     color_discrete_map=SEG_COLORS, points="outliers",
                     labels={sel: metrics[sel], "segment_label": "Segment"})
        fig.update_layout(showlegend=False)
        theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Normalised feature averages per segment (0–1 scale).")
        radar_cols = ["clv_90d", "prob_alive", "frequency", "monetary_avg",
                      "unique_products", "purchase_velocity", "tenure_days"]
        radar_labels = ["CLV", "P(Alive)", "Frequency", "Avg Order",
                        "Uniq Products", "Purchase Vel.", "Tenure"]
        fig = go.Figure()
        for seg, col in SEG_COLORS.items():
            row = df_full[df_full["segment_label"] == seg][radar_cols].mean()
            norms = []
            for c in radar_cols:
                mn, mx = df_full[c].min(), df_full[c].max()
                norms.append((row[c] - mn) / (mx - mn + 1e-9))
            norms.append(norms[0])  # close loop
            labels = radar_labels + [radar_labels[0]]
            fig.add_trace(go.Scatterpolar(r=norms, theta=labels, name=seg,
                                          line=dict(color=col, width=2),
                                          fill="toself", fillcolor=col,
                                          opacity=0.15))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                       tickfont=dict(size=9), gridcolor="#e2e8f0"),
                       angularaxis=dict(tickfont=dict(size=11), gridcolor="#e2e8f0")),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.12),
            height=450, margin=dict(t=20, b=60, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        heat_cols = ["clv_90d", "prob_alive", "frequency", "monetary_total",
                     "monetary_avg", "unique_products", "recency_days",
                     "purchase_velocity", "tenure_days", "predicted_purchases_90d"]
        heat_labels = ["CLV", "P(Alive)", "Frequency", "Total Spend", "Avg Order",
                       "Uniq Products", "Recency", "Velocity", "Tenure", "Pred Purch"]
        heat_data = []
        for seg in SEG_COLORS:
            row = df_full[df_full["segment_label"] == seg][heat_cols].mean().values
            heat_data.append(row)
        heat_arr = np.array(heat_data)
        # Min-max normalise per column
        col_min = heat_arr.min(axis=0)
        col_max = heat_arr.max(axis=0)
        heat_norm = (heat_arr - col_min) / (col_max - col_min + 1e-9)
        fig = go.Figure(go.Heatmap(
            z=heat_norm,
            x=heat_labels,
            y=list(SEG_COLORS.keys()),
            colorscale="RdYlGn",
            text=np.round(heat_arr, 1),
            texttemplate="%{text}",
            hovertemplate="<b>%{y} – %{x}</b><br>Raw: %{text}<br>Norm: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
            height=300, margin=dict(t=10, b=80, l=120, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        dims = ["clv_90d", "prob_alive", "frequency", "recency_days", "monetary_total"]
        fig = px.scatter_matrix(
            df_full.sample(min(1500, len(df_full)), random_state=42),
            dimensions=dims, color="segment_label",
            color_discrete_map=SEG_COLORS,
            labels={d: d.replace("_", " ").title() for d in dims},
            opacity=0.5,
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(height=520, paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 3 – CLV ANALYSIS
# =============================================================================

elif page == "💰 CLV Analysis":
    require_data()
    st.title("💰 Customer Lifetime Value Analysis")
    st.markdown("Understand the shape, drivers, and distribution of 90-day CLV across your portfolio.")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    p10, p25, p50, p75, p90, p99 = (
        df_full["clv_90d"].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.99]).values
    )
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("P10 CLV",  f"£{p10:,.2f}")
    k2.metric("P25 CLV",  f"£{p25:,.2f}")
    k3.metric("Median",   f"£{p50:,.2f}")
    k4.metric("P75 CLV",  f"£{p75:,.2f}")
    k5.metric("P90 CLV",  f"£{p90:,.2f}")
    k6.metric("P99 CLV",  f"£{p99:,.2f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Distribution", "🔗 CLV Drivers", "📉 Pareto", "💹 Revenue Mix", "📦 Percentile Bands"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cap = df_full["clv_90d"].quantile(0.99)
            fig = px.histogram(df_full[df_full["clv_90d"] < cap], x="clv_90d",
                               nbins=80, color="segment_label",
                               color_discrete_map=SEG_COLORS, barmode="overlay",
                               labels={"clv_90d": "CLV (£)", "segment_label": "Segment"})
            fig.update_traces(opacity=0.72)
            theme(fig, "CLV Histogram (capped at P99)", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.violin(df_full[df_full["clv_90d"] < cap], x="segment_label",
                            y="clv_90d", color="segment_label",
                            color_discrete_map=SEG_COLORS, box=True, points=False,
                            labels={"clv_90d": "CLV (£)", "segment_label": "Segment"})
            fig.update_layout(showlegend=False)
            theme(fig, "CLV Violin by Segment", height=380)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        driver_pairs = [
            ("frequency",        "clv_90d", "Frequency", "CLV (£)"),
            ("monetary_avg",     "clv_90d", "Avg Order Value (£)", "CLV (£)"),
            ("monetary_total",   "clv_90d", "Total Spend (£)", "CLV (£)"),
            ("prob_alive",       "clv_90d", "P(Alive)", "CLV (£)"),
            ("tenure_days",      "clv_90d", "Tenure (days)", "CLV (£)"),
            ("purchase_velocity","clv_90d", "Purchase Velocity", "CLV (£)"),
        ]
        sel_driver = st.selectbox(
            "X-axis driver",
            [d[0] for d in driver_pairs],
            format_func=lambda k: next(d[2] for d in driver_pairs if d[0] == k),
        )
        dp = next(d for d in driver_pairs if d[0] == sel_driver)
        cap_x = df_full[dp[0]].quantile(0.99)
        cap_y = df_full["clv_90d"].quantile(0.99)
        plot_df = df_full[(df_full[dp[0]] < cap_x) & (df_full["clv_90d"] < cap_y)]
        fig = px.scatter(plot_df, x=dp[0], y=dp[1],
                         color="segment_label", color_discrete_map=SEG_COLORS,
                         opacity=0.55,
                         labels={dp[0]: dp[2], dp[1]: dp[3], "segment_label": "Segment"})
        theme(fig, f"CLV vs. {dp[2]}", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sorted_clv = df_full["clv_90d"].sort_values(ascending=False).values
        cum_pct_cust = np.arange(1, len(sorted_clv) + 1) / len(sorted_clv) * 100
        cum_pct_rev  = np.cumsum(sorted_clv) / sorted_clv.sum() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_pct_cust, y=cum_pct_rev, mode="lines",
                                 line=dict(color="#6366f1", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
                                 name="Cumulative Revenue"))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                 line=dict(color="#94a3b8", width=1.5, dash="dash"),
                                 name="Perfect equality"))
        fig.add_vline(x=20, line_dash="dot", line_color="#EF4444", line_width=1.5,
                      annotation_text="Top 20%", annotation_position="top right")
        idx20 = np.searchsorted(cum_pct_cust, 20)
        rev20 = cum_pct_rev[min(idx20, len(cum_pct_rev)-1)]
        fig.add_annotation(x=20, y=rev20, text=f"→ {rev20:.1f}% revenue",
                           showarrow=True, arrowhead=2, ax=60, ay=-40,
                           font=dict(color="#EF4444", size=12))
        theme(fig, "Pareto Chart: Revenue Concentration", height=420)
        fig.update_xaxes(title="% of Customers")
        fig.update_yaxes(title="Cumulative % Revenue")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        rev_mix = df_full.groupby("segment_label")["clv_90d"].sum().reset_index()
        rev_mix.columns = ["Segment", "Total CLV (£)"]
        rev_mix["% Share"] = rev_mix["Total CLV (£)"] / rev_mix["Total CLV (£)"].sum() * 100
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(rev_mix.sort_values("Total CLV (£)", ascending=True),
                         x="Total CLV (£)", y="Segment", orientation="h",
                         color="Segment", color_discrete_map=SEG_COLORS,
                         text="Total CLV (£)")
            fig.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
            fig.update_layout(showlegend=False)
            theme(fig, "Total Portfolio CLV by Segment", height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(rev_mix, names="Segment", values="Total CLV (£)",
                         color="Segment", color_discrete_map=SEG_COLORS, hole=0.5)
            fig.update_traces(textinfo="percent+label")
            theme(fig, "Revenue Share by Segment", height=340)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        bands = [0, 10, 25, 50, 75, 90, 100]
        labels_b = [f"P{bands[i]}–P{bands[i+1]}" for i in range(len(bands)-1)]
        quantiles = [df_full["clv_90d"].quantile(b/100) for b in bands]
        band_data = []
        for i in range(len(labels_b)):
            mask = (df_full["clv_90d"] >= quantiles[i]) & (df_full["clv_90d"] < quantiles[i+1])
            n = mask.sum()
            rev = df_full.loc[mask, "clv_90d"].sum()
            band_data.append({"Band": labels_b[i], "Customers": n,
                               "Total CLV (£)": rev, "Avg CLV (£)": rev / max(n, 1)})
        band_df = pd.DataFrame(band_data)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(band_df, x="Band", y="Customers", color="Band",
                         color_discrete_sequence=px.colors.sequential.Viridis)
            theme(fig, "Customers per CLV Percentile Band", height=340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(band_df, x="Band", y="Total CLV (£)", color="Band",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            theme(fig, "Revenue per CLV Percentile Band", height=340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            band_df.style.format({"Total CLV (£)": "£{:,.0f}", "Avg CLV (£)": "£{:,.2f}"}),
            use_container_width=True,
        )


# =============================================================================
# PAGE 4 – CHURN ANALYSIS
# =============================================================================

elif page == "⚠️ Churn Analysis":
    require_data()
    st.title("⚠️ Churn & Retention Analysis")
    st.markdown("Monitor churn probability, P(Alive), and identify at-risk cohorts before they lapse.")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    churn_rate   = df_full["churned"].mean() * 100 if "churned" in df_full.columns else 0
    active_rate  = 100 - churn_rate
    high_risk    = (df_full["churn_proba"] > 0.6).sum() if "churn_proba" in df_full.columns else 0
    avg_churn_p  = df_full["churn_proba"].mean() * 100 if "churn_proba" in df_full.columns else 0
    dead_likely  = (df_full["prob_alive"] < 0.3).sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Overall Churn Rate",      f"{churn_rate:.1f}%")
    k2.metric("Active Rate",             f"{active_rate:.1f}%")
    k3.metric("High-Risk Customers",     f"{high_risk:,}")
    k4.metric("Avg Churn Probability",   f"{avg_churn_p:.1f}%")
    k5.metric("P(Alive) < 30%",          f"{dead_likely:,}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distributions", "🎯 Risk Matrix", "📈 Recency Analysis", "🔥 High-Risk Customers"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "churn_proba" in df_full.columns:
                fig = px.histogram(df_full, x="churn_proba", color="segment_label",
                                   nbins=50, barmode="overlay",
                                   color_discrete_map=SEG_COLORS,
                                   labels={"churn_proba": "Churn Probability",
                                           "segment_label": "Segment"})
                fig.update_traces(opacity=0.7)
                fig.add_vline(x=0.6, line_dash="dash", line_color="#EF4444",
                              annotation_text="High-risk threshold (0.6)",
                              annotation_position="top right")
                theme(fig, "Churn Probability Distribution", height=380)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df_full, x="prob_alive", color="segment_label",
                               nbins=50, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"prob_alive": "P(Alive)", "segment_label": "Segment"})
            fig.update_traces(opacity=0.7)
            fig.add_vline(x=0.3, line_dash="dash", line_color="#EF4444",
                          annotation_text="Low vitality (<0.3)",
                          annotation_position="top left")
            theme(fig, "P(Alive) Distribution", height=380)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "churned" in df_full.columns:
                fig = px.box(df_full, x="churned", y="churn_proba" if "churn_proba" in df_full.columns else "prob_alive",
                             color="churned", points=False,
                             color_discrete_map={0: "#10B981", 1: "#EF4444"},
                             labels={"churned": "Churned", "churn_proba": "Churn Probability"})
                fig.update_layout(showlegend=False)
                theme(fig, "Churn Prob: Active vs. Churned", height=320)
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "churned" in df_full.columns:
                churn_seg = df_full.groupby(["segment_label", "churned"]).size().reset_index(name="count")
                churn_seg["Status"] = churn_seg["churned"].map({0: "Active", 1: "Churned"})
                fig = px.bar(churn_seg, x="segment_label", y="count", color="Status",
                             barmode="stack",
                             color_discrete_map={"Active": "#10B981", "Churned": "#EF4444"},
                             labels={"segment_label": "Segment", "count": "Customers"})
                theme(fig, "Active vs. Churned by Segment (Stacked)", height=320)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Each dot is a customer. Quadrants show retention priority.")
        cap_r = df_full["recency_days"].quantile(0.98)
        plot_df = df_full[df_full["recency_days"] < cap_r].copy()
        churn_col = "churn_proba" if "churn_proba" in plot_df.columns else "prob_alive"
        fig = px.scatter(plot_df.sample(min(2000, len(plot_df)), random_state=1),
                         x="prob_alive", y=churn_col,
                         color="segment_label", color_discrete_map=SEG_COLORS,
                         size="clv_90d", size_max=18, opacity=0.6,
                         labels={"prob_alive": "P(Alive)", churn_col: "Churn Probability",
                                 "clv_90d": "CLV (£)", "segment_label": "Segment"})
        fig.add_hline(y=0.6, line_dash="dot", line_color="#EF4444", line_width=1.5)
        fig.add_vline(x=0.5, line_dash="dot", line_color="#F59E0B", line_width=1.5)
        fig.add_annotation(x=0.2, y=0.9, text="🔴 High Risk", showarrow=False,
                           font=dict(color="#EF4444", size=13, family="Inter"))
        fig.add_annotation(x=0.8, y=0.2, text="🟢 Safe", showarrow=False,
                           font=dict(color="#10B981", size=13, family="Inter"))
        theme(fig, "Churn Risk Matrix", height=440)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cap_r2 = df_full["recency_days"].quantile(0.98)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_full[df_full["recency_days"] < cap_r2],
                               x="recency_days", color="segment_label",
                               nbins=60, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"recency_days": "Days Since Last Purchase",
                                       "segment_label": "Segment"})
            fig.update_traces(opacity=0.7)
            theme(fig, "Recency Distribution by Segment", height=360)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                df_full[(df_full["recency_days"] < cap_r2)].sample(min(2000, len(df_full)), random_state=2),
                x="recency_days", y="prob_alive",
                color="segment_label", color_discrete_map=SEG_COLORS,
                opacity=0.5,
                labels={"recency_days": "Days Since Last Purchase", "prob_alive": "P(Alive)",
                        "segment_label": "Segment"},
            )
            theme(fig, "Recency vs. P(Alive)", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        churn_threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.6, 0.05)
        if "churn_proba" in df_full.columns:
            at_risk = df_full[df_full["churn_proba"] > churn_threshold].copy()
        else:
            at_risk = df_full[df_full["prob_alive"] < (1 - churn_threshold)].copy()
        at_risk = at_risk.sort_values("clv_90d", ascending=False)
        show_cols = [c for c in ["customer_id", "segment_label", "clv_90d", "churn_proba",
                                  "prob_alive", "recency_days", "frequency", "monetary_total"]
                     if c in at_risk.columns]
        st.markdown(f"**{len(at_risk):,}** customers above threshold")
        st.dataframe(
            at_risk[show_cols].reset_index(drop=True).style.format({
                "clv_90d":      "£{:.2f}",
                "churn_proba":  "{:.1%}",
                "prob_alive":   "{:.1%}",
                "recency_days": "{:.0f}",
                "monetary_total": "£{:,.0f}",
            }),
            use_container_width=True,
            height=420,
        )
        # Download
        buf = io.StringIO()
        at_risk[show_cols].to_csv(buf, index=False)
        st.download_button("⬇️ Download at-risk list (CSV)", buf.getvalue(),
                           "at_risk_customers.csv", "text/csv")


# =============================================================================
# PAGE 5 – RFM ANALYSIS
# =============================================================================

elif page == "📐 RFM Analysis":
    require_data()
    st.title("📐 RFM Analysis")
    st.markdown("Recency · Frequency · Monetary — the three pillars of customer behaviour.")
    st.markdown("---")

    rfm_cols = ["recency_days", "frequency", "monetary_total", "monetary_avg"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Recency (days)",     f"{df_full['recency_days'].mean():.0f}")
    k2.metric("Avg Frequency",          f"{df_full['frequency'].mean():.1f}")
    k3.metric("Avg Total Spend",        f"£{df_full['monetary_total'].mean():,.0f}")
    k4.metric("Avg Order Value",        f"£{df_full['monetary_avg'].mean():,.2f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Individual Distributions", "🔴 3-D RFM", "🌡️ Correlation", "📅 Purchase Patterns"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cap_r = df_full["recency_days"].quantile(0.98)
            fig = px.histogram(df_full[df_full["recency_days"] < cap_r],
                               x="recency_days", color="segment_label",
                               nbins=60, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"recency_days": "Days Since Last Purchase",
                                       "segment_label": "Segment"})
            fig.update_traces(opacity=0.72)
            theme(fig, "Recency Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cap_f = df_full["frequency"].quantile(0.98)
            fig = px.histogram(df_full[df_full["frequency"] < cap_f],
                               x="frequency", color="segment_label",
                               nbins=50, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"frequency": "Purchase Frequency",
                                       "segment_label": "Segment"})
            fig.update_traces(opacity=0.72)
            theme(fig, "Frequency Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            cap_m = df_full["monetary_total"].quantile(0.99)
            fig = px.histogram(df_full[df_full["monetary_total"] < cap_m],
                               x="monetary_total", color="segment_label",
                               nbins=60, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"monetary_total": "Total Spend (£)",
                                       "segment_label": "Segment"})
            fig.update_traces(opacity=0.72)
            theme(fig, "Monetary (Total Spend) Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            cap_ma = df_full["monetary_avg"].quantile(0.99)
            fig = px.histogram(df_full[df_full["monetary_avg"] < cap_ma],
                               x="monetary_avg", color="segment_label",
                               nbins=60, barmode="overlay",
                               color_discrete_map=SEG_COLORS,
                               labels={"monetary_avg": "Avg Order Value (£)",
                                       "segment_label": "Segment"})
            fig.update_traces(opacity=0.72)
            theme(fig, "Average Order Value Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("3-D scatter: Recency · Frequency · Monetary – colour = segment.")
        sample = df_full.sample(min(2000, len(df_full)), random_state=7)
        cap_m3 = sample["monetary_total"].quantile(0.99)
        sample = sample[sample["monetary_total"] < cap_m3]
        fig = px.scatter_3d(sample, x="recency_days", y="frequency", z="monetary_total",
                            color="segment_label", color_discrete_map=SEG_COLORS,
                            opacity=0.65, size_max=5,
                            labels={"recency_days": "Recency (days)",
                                    "frequency": "Frequency",
                                    "monetary_total": "Total Spend (£)",
                                    "segment_label": "Segment"})
        fig.update_layout(height=560, paper_bgcolor="white",
                          scene=dict(
                              xaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
                              yaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
                              zaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
                          ))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = ["recency_days", "frequency", "monetary_total", "monetary_avg",
                    "unique_products", "tenure_days", "purchase_velocity",
                    "avg_basket_size", "avg_items_per_order", "clv_90d", "prob_alive"]
        num_cols = [c for c in num_cols if c in df_full.columns]
        corr = df_full[num_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=480, paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(t=20, b=80, l=120, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if "preferred_dow" in df_full.columns:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                dow_counts = df_full["preferred_dow"].value_counts().sort_index()
                fig = px.bar(x=[day_names[i] for i in dow_counts.index],
                             y=dow_counts.values, color=dow_counts.values,
                             color_continuous_scale="Blues",
                             labels={"x": "Day of Week", "y": "Customers"})
                fig.update_coloraxes(showscale=False)
                fig.update_layout(showlegend=False)
                theme(fig, "Preferred Purchase Day", height=320)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "preferred_hour" in df_full.columns:
                hour_counts = df_full["preferred_hour"].value_counts().sort_index()
                fig = px.bar(x=hour_counts.index, y=hour_counts.values,
                             color=hour_counts.values,
                             color_continuous_scale="Purples",
                             labels={"x": "Hour of Day", "y": "Customers"})
                fig.update_coloraxes(showscale=False)
                fig.update_layout(showlegend=False)
                theme(fig, "Preferred Purchase Hour", height=320)
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "tenure_days" in df_full.columns:
                cap_t = df_full["tenure_days"].quantile(0.99)
                fig = px.histogram(df_full[df_full["tenure_days"] < cap_t],
                                   x="tenure_days", color="segment_label",
                                   nbins=50, barmode="overlay",
                                   color_discrete_map=SEG_COLORS,
                                   labels={"tenure_days": "Tenure (days)",
                                           "segment_label": "Segment"})
                fig.update_traces(opacity=0.72)
                theme(fig, "Customer Tenure Distribution", height=320)
                st.plotly_chart(fig, use_container_width=True)
        with col4:
            if "unique_products" in df_full.columns:
                cap_u = df_full["unique_products"].quantile(0.99)
                fig = px.histogram(df_full[df_full["unique_products"] < cap_u],
                                   x="unique_products", color="segment_label",
                                   nbins=50, barmode="overlay",
                                   color_discrete_map=SEG_COLORS,
                                   labels={"unique_products": "Unique Products Purchased",
                                           "segment_label": "Segment"})
                fig.update_traces(opacity=0.72)
                theme(fig, "Product Diversity Distribution", height=320)
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 6 – CUSTOMER EXPLORER
# =============================================================================

elif page == "🔎 Customer Explorer":
    require_data()
    st.title("🔎 Customer Explorer")
    st.markdown("Filter, search, and drill into individual customers. Download any subset.")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🎛️ Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            seg_filter = st.multiselect(
                "Segment", options=sorted(df_full["segment_label"].unique()),
                default=sorted(df_full["segment_label"].unique()),
            )
        with fc2:
            clv_min, clv_max = float(df_full["clv_90d"].min()), float(df_full["clv_90d"].quantile(0.99))
            clv_range = st.slider("CLV Range (£)", clv_min, clv_max,
                                  (clv_min, clv_max), step=1.0)
        with fc3:
            rec_min, rec_max = int(df_full["recency_days"].min()), int(df_full["recency_days"].quantile(0.99))
            rec_range = st.slider("Recency (days)", rec_min, rec_max,
                                  (rec_min, rec_max), step=1)

        fc4, fc5 = st.columns(2)
        with fc4:
            if "churn_proba" in df_full.columns:
                churn_range = st.slider("Churn Probability", 0.0, 1.0, (0.0, 1.0), 0.01)
            else:
                churn_range = (0.0, 1.0)
        with fc5:
            palive_range = st.slider("P(Alive)", 0.0, 1.0, (0.0, 1.0), 0.01)

    # Apply filters
    mask = (
        df_full["segment_label"].isin(seg_filter)
        & df_full["clv_90d"].between(*clv_range)
        & df_full["recency_days"].between(*rec_range)
        & df_full["prob_alive"].between(*palive_range)
    )
    if "churn_proba" in df_full.columns:
        mask &= df_full["churn_proba"].between(*churn_range)

    filtered = df_full[mask].copy()
    st.markdown(f"**{len(filtered):,}** customers match your filters")
    st.markdown("---")

    # ── Mini KPIs ──────────────────────────────────────────────────────────────
    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Customers",    f"{len(filtered):,}")
    mk2.metric("Total CLV",    f"£{filtered['clv_90d'].sum():,.0f}")
    mk3.metric("Mean CLV",     f"£{filtered['clv_90d'].mean():,.2f}")
    mk4.metric("Avg P(Alive)", f"{filtered['prob_alive'].mean():.1%}")

    st.markdown("---")

    # ── Quick charts ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        seg_f = filtered["segment_label"].value_counts().reset_index()
        seg_f.columns = ["Segment", "Count"]
        fig = px.bar(seg_f, x="Segment", y="Count", color="Segment",
                     color_discrete_map=SEG_COLORS)
        fig.update_layout(showlegend=False)
        theme(fig, "Segment Breakdown (filtered)", height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        cap_f = filtered["clv_90d"].quantile(0.99) if len(filtered) > 10 else filtered["clv_90d"].max()
        fig = px.histogram(filtered[filtered["clv_90d"] <= cap_f],
                           x="clv_90d", color="segment_label",
                           nbins=40, barmode="overlay",
                           color_discrete_map=SEG_COLORS,
                           labels={"clv_90d": "CLV (£)", "segment_label": "Segment"})
        fig.update_traces(opacity=0.72)
        theme(fig, "CLV Distribution (filtered)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Table ─────────────────────────────────────────────────────────────────
    display_cols = [c for c in ["customer_id", "segment_label", "clv_90d",
                                 "prob_alive", "churn_proba", "churned",
                                 "predicted_purchases_90d", "recency_days",
                                 "frequency", "monetary_total", "monetary_avg",
                                 "unique_products", "tenure_days"]
                    if c in filtered.columns]
    sort_col = st.selectbox("Sort by", options=display_cols,
                             index=display_cols.index("clv_90d") if "clv_90d" in display_cols else 0)
    asc = st.checkbox("Ascending", value=False)
    show_df = filtered[display_cols].sort_values(sort_col, ascending=asc).reset_index(drop=True)

    fmt = {
        "clv_90d": "£{:.2f}", "prob_alive": "{:.1%}",
        "predicted_purchases_90d": "{:.2f}", "recency_days": "{:.0f}",
        "frequency": "{:.0f}", "monetary_total": "£{:,.0f}",
        "monetary_avg": "£{:,.2f}",
    }
    if "churn_proba" in show_df.columns:
        fmt["churn_proba"] = "{:.1%}"

    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=440)

    # ── Download ──────────────────────────────────────────────────────────────
    buf = io.StringIO()
    show_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download filtered customers (CSV)", buf.getvalue(),
                       "filtered_customers.csv", "text/csv")


# =============================================================================
# PAGE 7 – SCORE A CUSTOMER
# =============================================================================

elif page == "🚀 Score a Customer":
    st.title("🚀 Score a Customer")
    st.markdown("Enter a customer's features to get real-time CLV, churn probability, uplift score, and segment.")
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-header">Purchase History</p>', unsafe_allow_html=True)
        bgf_frequency  = st.number_input("Repeat Purchases (frequency)",             min_value=1,   max_value=500,  value=5)
        bgf_recency    = st.number_input("Days: First → Last Purchase (recency)",     min_value=1,   max_value=1000, value=200)
        bgf_T          = st.number_input("Customer Age in Days (T)",                  min_value=1,   max_value=1000, value=365)
        frequency      = st.number_input("Total Invoices",                            min_value=1,   max_value=500,  value=6)
        recency_days   = st.number_input("Days Since Last Purchase",                  min_value=0,   max_value=1000, value=30)

    with col2:
        st.markdown('<p class="section-header">Spending Behaviour</p>', unsafe_allow_html=True)
        monetary_total      = st.number_input("Total Spend (£)",           min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
        monetary_avg        = st.number_input("Avg Order Value (£)",       min_value=0.0, max_value=10000.0,  value=100.0, step=5.0)
        avg_basket_size     = st.number_input("Avg Basket Size (units)",   min_value=0.0, max_value=1000.0,   value=12.0,  step=1.0)
        avg_items_per_order = st.number_input("Avg Items per Order",       min_value=0.0, max_value=500.0,    value=8.0,   step=1.0)
        unique_products     = st.number_input("Unique Products Purchased", min_value=1,   max_value=2000,     value=20)

    with col3:
        st.markdown('<p class="section-header">Temporal Patterns</p>', unsafe_allow_html=True)
        tenure_days        = st.number_input("Tenure (days)",              min_value=0,   max_value=1000,  value=300)
        purchase_velocity  = st.number_input("Avg Days Between Orders",    min_value=0.0, max_value=500.0, value=30.0, step=1.0)
        predicted_90d      = st.number_input("Predicted Purchases (90d)",  min_value=0.0, max_value=50.0,  value=1.5,  step=0.1)
        preferred_dow      = st.selectbox("Preferred Day of Week", options=[0, 1, 2, 3, 4, 5, 6],
                                          format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        preferred_hour     = st.slider("Preferred Hour of Day", min_value=0, max_value=23, value=11)

    st.markdown("---")

    if st.button("🚀 Score Customer", type="primary", use_container_width=True):

        ml_features = {
            "recency_days":           recency_days,
            "frequency":              frequency,
            "monetary_total":         monetary_total,
            "monetary_avg":           monetary_avg,
            "tenure_days":            tenure_days,
            "purchase_velocity":      purchase_velocity,
            "unique_products":        unique_products,
            "avg_basket_size":        avg_basket_size,
            "avg_items_per_order":    avg_items_per_order,
            "predicted_purchases_90d": predicted_90d,
            "preferred_dow":          preferred_dow,
            "preferred_hour":         preferred_hour,
        }
        X = pd.DataFrame([ml_features])

        # BG/NBD
        prob_alive = float(bgf.conditional_probability_alive(bgf_frequency, bgf_recency, bgf_T))
        predicted_purchases = float(bgf.conditional_expected_number_of_purchases_up_to_time(
            90, bgf_frequency, bgf_recency, bgf_T
        ))

        # Gamma-Gamma CLV
        clv = ggf.customer_lifetime_value(
            bgf,
            pd.Series([bgf_frequency]),
            pd.Series([bgf_recency]),
            pd.Series([bgf_T]),
            pd.Series([monetary_avg]),
            time=3,
            discount_rate=0.01,
        ).values[0] * 0.20

        churn_prob   = float(churn_model.predict_proba(X)[0, 1])
        uplift_score = float(uplift_model.predict(X)[0])
        segment      = assign_segment_label(prob_alive, monetary_total, churn_prob)

        # ── Results KPIs ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Scoring Results")

        seg_icon = {"Champions": "🟢", "Loyal Customers": "🔵",
                    "At Risk": "🟠", "Lost / Dormant": "🔴"}

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Segment",             f"{seg_icon.get(segment,'')} {segment}")
        r2.metric("90-Day CLV",          f"£{float(clv):,.2f}")
        r3.metric("P(Alive)",            f"{prob_alive:.1%}")
        r4.metric("Churn Probability",   f"{churn_prob:.1%}")
        r5.metric("Predicted Purchases", f"{predicted_purchases:.2f}")

        st.markdown("---")

        # ── Gauge charts ──────────────────────────────────────────────────────
        def make_gauge(value, title, color, reverse=False):
            pct = value * 100
            if reverse:
                bar_color = "#10B981" if pct < 33 else "#F59E0B" if pct < 66 else "#EF4444"
            else:
                bar_color = "#EF4444" if pct < 33 else "#F59E0B" if pct < 66 else "#10B981"
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                title={"text": title, "font": {"size": 14, "color": "#334155"}},
                number={"suffix": "%", "font": {"size": 24, "color": "#1e293b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar":  {"color": bar_color, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  33], "color": "#fef2f2"},
                        {"range": [33, 66], "color": "#fffbeb"},
                        {"range": [66, 100], "color": "#f0fdf4"},
                    ],
                    "threshold": {"line": {"color": "#1e293b", "width": 2},
                                  "thickness": 0.75, "value": pct},
                },
            ))
            fig.update_layout(height=260, paper_bgcolor="white",
                              margin=dict(t=50, b=10, l=20, r=20))
            return fig

        cg1, cg2, cg3 = st.columns(3)
        cg1.plotly_chart(make_gauge(prob_alive, "P(Alive)", "#10B981", reverse=False),
                         use_container_width=True)
        cg2.plotly_chart(make_gauge(churn_prob, "Churn Probability", "#EF4444", reverse=True),
                         use_container_width=True)
        cg3.plotly_chart(make_gauge(min(abs(uplift_score), 1.0), "Uplift Score", "#6366f1", reverse=False),
                         use_container_width=True)

        st.markdown("---")

        # ── Feature breakdown bar ─────────────────────────────────────────────
        st.subheader("🔍 Input Feature Summary")
        feat_df = pd.DataFrame({"Feature": list(ml_features.keys()),
                                 "Value":   list(ml_features.values())})
        feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ").str.title()

        # Normalise for comparison bar
        if data_loaded:
            normed = []
            for k, v in ml_features.items():
                if k in df_full.columns:
                    mn, mx = df_full[k].min(), df_full[k].max()
                    normed.append((v - mn) / (mx - mn + 1e-9))
                else:
                    normed.append(0.5)
            feat_df["Percentile (approx)"] = np.clip(normed, 0, 1)
            fig_feat = px.bar(feat_df, x="Percentile (approx)", y="Feature",
                              orientation="h", color="Percentile (approx)",
                              color_continuous_scale="RdYlGn",
                              labels={"Percentile (approx)": "Relative to Portfolio (0=lowest, 1=highest)"},
                              text="Value")
            fig_feat.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_feat.update_coloraxes(showscale=False)
            theme(fig_feat, height=400)
            st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("---")

        # ── Recommendation ────────────────────────────────────────────────────
        st.subheader("💡 Recommended Action")

        reco_cfg = {
            "Champions": {
                "color": "#f0fdf4", "border": "#10B981",
                "icon": "🟢",
                "title": "Retain & Reward",
                "text": "Top-value customer. Offer exclusive loyalty rewards, early product access, or VIP membership. Avoid generic promotions — personalise every touchpoint to deepen the relationship and prevent competitive poaching.",
            },
            "Loyal Customers": {
                "color": "#eff6ff", "border": "#3B82F6",
                "icon": "🔵",
                "title": "Nurture & Grow",
                "text": "Solid, consistent buyer with real growth potential. Deploy personalised product recommendations based on purchase history and moderate incentives to increase order frequency and basket size.",
            },
            "At Risk": {
                "color": "#fffbeb", "border": "#F59E0B",
                "icon": "🟠",
                "title": "Win-Back Campaign",
                "text": "This customer is drifting. Launch a targeted re-engagement offer — time-limited discount or personalised bundle — within the next 2 weeks. Monitor response and escalate to a loyalty incentive if no engagement.",
            },
            "Lost / Dormant": {
                "color": "#fef2f2", "border": "#EF4444",
                "icon": "🔴",
                "title": "Low-Cost Reactivation",
                "text": "High churn probability detected. Attempt a single low-cost bulk reactivation email with a compelling offer. If no response within 30 days, deprioritise this customer to protect marketing ROI.",
            },
        }
        cfg = reco_cfg.get(segment, reco_cfg["Lost / Dormant"])
        st.markdown(f"""
        <div style="background:{cfg['color']};border-left:5px solid {cfg['border']};
                    border-radius:12px;padding:1.3rem 1.6rem;margin-top:0.5rem;">
          <div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:.4rem;">
            {cfg['icon']} {cfg['title']}
          </div>
          <div style="font-size:0.92rem;color:#334155;line-height:1.7;">
            {cfg['text']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Comparator: how does this customer compare to their segment? ───────
        if data_loaded:
            st.markdown("---")
            st.subheader(f"📊 Customer vs. {segment} Segment Average")
            seg_avg = df_full[df_full["segment_label"] == segment][list(ml_features.keys())].mean()
            compare_data = pd.DataFrame({
                "Feature": [k.replace("_", " ").title() for k in ml_features],
                "This Customer": list(ml_features.values()),
                "Segment Avg":   seg_avg.values,
            })
            compare_long = compare_data.melt(id_vars="Feature", var_name="Source", value_name="Value")
            fig_cmp = px.bar(compare_long, x="Value", y="Feature", color="Source",
                             barmode="group", orientation="h",
                             color_discrete_map={"This Customer": "#6366f1",
                                                  "Segment Avg": "#94a3b8"})
            theme(fig_cmp, height=420)
            st.plotly_chart(fig_cmp, use_container_width=True)
