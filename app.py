"""
Shopper Spectrum - Streamlit Application
Customer Segmentation and Product Recommendations in E-Commerce

This app has two core modules, exactly as specified in the project brief:

    1. Product Recommendation - takes a product name, returns 5 similar products
    2. Customer Segmentation  - takes Recency, Frequency, Monetary, returns the predicted segment

Plus supporting analytics views built on the same saved artifacts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# Page Configuration
st.set_page_config(

    page_title = "Shopper Spectrum",

    page_icon = "🛒",

    layout = "wide",

    initial_sidebar_state = "expanded"

)


# Theme state — light / dark

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

_resolved_theme = st.session_state["theme"]


# Two full palettes as plain Python dicts. The active one is picked in Python
# (from st.session_state) and its values are written straight onto :root as
# CSS variables — no JavaScript, no cross-document / iframe attribute setting.
# This is what makes the toggle reliably work inside Streamlit's app, including
# the sandboxed component iframes used by some widgets.

THEMES = {

    "dark": {

        "bg-0": "#0b0e11", "bg-1": "#12161b", "bg-2": "#181d24", "line": "#242b33",

        "text-0": "#eef1f4", "text-1": "#9aa5b1", "text-2": "#5f6b78",

        "accent": "#2dd4bf", "accent-dim": "#12403c", "accent-ink": "#05221e",

        "warn": "#f5a623", "danger": "#ef4b4b", "grid": "#242b33",

        "shadow": "rgba(45, 212, 191, 0.35)", "shadow-strong": "rgba(45, 212, 191, 0.55)",

        "glow-r1": "radial-gradient(circle at 15% 0%, #10261f 0%, transparent 35%)",

        "glow-r2": "radial-gradient(circle at 100% 20%, #1a1420 0%, transparent 40%)",

        "color-scheme": "dark"

    },

    "light": {

        "bg-0": "#f6f7f9", "bg-1": "#ffffff", "bg-2": "#ffffff", "line": "#e2e6eb",

        "text-0": "#14181d", "text-1": "#51606e", "text-2": "#8996a3",

        "accent": "#0f9d8f", "accent-dim": "#d7f3ef", "accent-ink": "#ffffff",

        "warn": "#b3720b", "danger": "#c8353a", "grid": "#e6e9ed",

        "shadow": "rgba(15, 157, 143, 0.18)", "shadow-strong": "rgba(15, 157, 143, 0.28)",

        "glow-r1": "radial-gradient(circle at 15% 0%, #e6f7f4 0%, transparent 40%)",

        "glow-r2": "radial-gradient(circle at 100% 20%, #f0eef9 0%, transparent 45%)",

        "color-scheme": "light"

    }

}

_active_theme = THEMES[_resolved_theme]

_theme_vars_css = "\n".join(f"--{k}: {v};" for k, v in _active_theme.items() if k != "color-scheme")


# Global Styling — theme-able console UI (dark by default, light on toggle)

st.markdown(

    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
{_theme_vars_css}
        color-scheme: {_active_theme["color-scheme"]};
    }}

    :root {{
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Inter', sans-serif;
    }}

    html, body, [class*="css"] {{
        font-family: var(--sans);
    }}

    .stApp {{
        background:
            var(--glow-r1),
            var(--glow-r2),
            var(--bg-0) !important;
        color: var(--text-0) !important;
    }}

    /* Streamlit's native theme system also paints these containers directly;
    force our palette to win regardless of DOM/stylesheet load order so a
    native Light/Dark switch doesn't leave stale dark panels behind. */
    [data-testid="stMain"], [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"], .main {{
        background-color: var(--bg-0) !important;
        color: var(--text-0) !important;
    }}

    /* This app has its own light/dark toggle, so Streamlit's native "⋮"
    menu (Settings → Theme) is redundant and hidden here to avoid two
    competing theme switches. Footer credit hidden as before. */
    footer {{visibility: hidden;}}

    #MainMenu {{visibility: hidden;}}

    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
        height: 3.2rem !important;
        min-height: 3.2rem !important;
        z-index: 999999 !important;
    }}

    div[data-testid="stToolbar"] {{
        visibility: hidden !important;
    }}

    /* ---------- Sidebar collapse / expand toggle ----------
    This is the button the USER clicks to hide/show the sidebar
    whenever they want. Made large + glowing so it's obvious
    and easy to find, in both collapsed and expanded states. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 999999 !important;
    }}

    button[data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stBaseButton-headerNoPadding"] {{
        visibility: visible !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        color: var(--accent) !important;
        background: var(--bg-2) !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        width: 42px !important;
        height: 42px !important;
        box-shadow: 0 0 14px var(--shadow) !important;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }}

    button[data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[data-testid="stBaseButton-headerNoPadding"]:hover {{
        box-shadow: 0 0 22px var(--shadow-strong) !important;
        transform: translateY(-1px);
    }}

    button[data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg {{
        fill: var(--accent) !important;
        color: var(--accent) !important;
        width: 22px !important;
        height: 22px !important;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: var(--bg-1) !important;
        border-right: 1px solid var(--line);
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-0);
    }}

    /* ---------- Headings ---------- */
    h1, h2, h3 {{
        font-family: var(--sans);
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-0) !important;
    }}

    .eyebrow {{
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
        display: block;
    }}

    .subtext {{
        color: var(--text-1);
        font-size: 0.98rem;
        line-height: 1.6;
    }}

    hr {{
        border-color: var(--line) !important;
    }}

    /* ---------- Metric / stat cards ---------- */
    .stat-card {{
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }}
    .stat-card .label {{
        font-family: var(--mono);
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-2);
        margin-bottom: 0.35rem;
        display: block;
    }}
    .stat-card .value {{
        font-family: var(--mono);
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-0);
        line-height: 1.1;
    }}
    .stat-card .delta {{
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--accent);
        margin-top: 0.3rem;
        display: block;
    }}

    /* ---------- Segment result banner ---------- */
    .segment-banner {{
        background: linear-gradient(135deg, var(--accent-dim), var(--bg-2));
        border: 1px solid var(--accent);
        border-radius: 12px;
        padding: 1.8rem 2rem;
        margin-top: 1rem;
    }}
    .segment-banner .tag {{
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
    }}
    .segment-banner .title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text-0);
        margin: 0.25rem 0 0.5rem 0;
    }}
    .segment-banner .desc {{
        color: var(--text-1);
        font-size: 0.98rem;
        line-height: 1.55;
    }}

    /* ---------- Recommendation cards ---------- */
    .rec-card {{
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        transition: border-color 0.15s ease;
    }}
    .rec-card:hover {{
        border-color: var(--accent);
    }}
    .rec-rank {{
        font-family: var(--mono);
        font-size: 0.85rem;
        color: var(--accent);
        background: var(--accent-dim);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        min-width: 2.1rem;
        text-align: center;
    }}
    .rec-name {{
        font-size: 0.95rem;
        color: var(--text-0);
        font-weight: 500;
    }}
    .rec-score {{
        margin-left: auto;
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--text-2);
    }}

    /* ---------- Badges ---------- */
    .badge {{
        display: inline-block;
        font-family: var(--mono);
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        color: var(--text-1);
        margin-right: 0.4rem;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        background: var(--accent);
        color: var(--accent-ink);
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-family: var(--sans);
        padding: 0.55rem 1.4rem;
        transition: transform 0.1s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px var(--shadow);
        color: var(--accent-ink);
    }}

    /* ---------- Inputs ---------- */
    .stTextInput input, .stNumberInput input {{
        background: var(--bg-2) !important;
        color: var(--text-0) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid var(--line);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: var(--mono);
        font-size: 0.82rem;
        color: var(--text-1);
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
    }}

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }}

    .footer-note {{
        font-family: var(--mono);
        font-size: 0.72rem;
        color: var(--text-2);
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }}

    </style>
    """,

    unsafe_allow_html = True
)


# Icon system — thin-stroke linear SVGs (Lucide-style), inherit currentColor
# so they automatically match text color / theme without extra CSS rules.
# Used in place of emoji for a clean, professional look throughout the app.

def icon(name, size = 22, stroke_width = 1.6, color = "currentColor"):

    "Return an inline SVG <svg> string for the given icon name."

    paths = {

        "storefront": (
            '<path d="M3 9l1.5-5h15L21 9"/>'
            '<path d="M3 9a2.5 2.5 0 0 0 5 0 2.5 2.5 0 0 0 5 0 2.5 2.5 0 0 0 5 0 2.5 2.5 0 0 0 5 0"/>'
            '<path d="M5 9v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9"/>'
            '<path d="M9.5 20v-5.5a1.5 1.5 0 0 1 1.5-1.5h2a1.5 1.5 0 0 1 1.5 1.5V20"/>'
        ),

        "target": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<circle cx="12" cy="12" r="5"/>'
            '<circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/>'
        ),

        "cart": (
            '<circle cx="9.5" cy="20" r="1.4"/>'
            '<circle cx="17.5" cy="20" r="1.4"/>'
            '<path d="M2.5 3h2.2l2.1 12.2a1.8 1.8 0 0 0 1.78 1.5h9.14a1.8 1.8 0 0 0 1.77-1.46L21.5 7H6.1"/>'
        ),

        "chart": (
            '<path d="M4 20V10"/>'
            '<path d="M11 20V4"/>'
            '<path d="M18 20v-7"/>'
            '<path d="M3 20h18"/>'
        ),

        "compass": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M15.2 8.8l-2 5.2-5.2 2 2-5.2z"/>'
        ),

        "sparkle": (
            '<path d="M12 3v3.2"/>'
            '<path d="M12 17.8V21"/>'
            '<path d="M3 12h3.2"/>'
            '<path d="M17.8 12H21"/>'
            '<path d="M5.6 5.6l2.3 2.3"/>'
            '<path d="M16.1 16.1l2.3 2.3"/>'
            '<path d="M5.6 18.4l2.3-2.3"/>'
            '<path d="M16.1 7.9l2.3-2.3"/>'
        ),

    }

    body = paths.get(name, "")

    return (
        
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        
        f'stroke-linejoin="round" style="vertical-align:-4px; display:inline-block;">{body}</svg>'
        
    )


# Segment metadata — business-facing descriptions + tone
SEGMENT_INFO = {

    "High-Value": {

        "desc": "Buys often, spends the most, and purchased recently. This is the core revenue group — prioritise retention, early access, and loyalty perks.",
        "color": "#2dd4bf"

    },

    "Regular": {

        "desc": "Consistent, moderate purchasing behaviour. Reliable revenue base with room to grow into High-Value through targeted upsell.",
        "color": "#5b9dd9"

    },

    "Occasional": {

        "desc": "Infrequent, lower-spend purchases. Responds well to promotions and reminders — a re-engagement campaign is usually the right lever.",
        "color": "#f5a623"

    },

    "At-Risk": {

        "desc": "Long time since last purchase. Highest churn probability in the base — needs a win-back offer before the relationship goes cold.",
        "color": "#ef4b4b"

    }

}


# Load Saved Models
@st.cache_resource
def load_models():

    "Load all saved model artifacts from the models/ directory."

    kmeans = joblib.load('models/kmeans_model.pkl')

    scaler = joblib.load('models/rfm_scaler.pkl')

    cluster_label_map = joblib.load('models/cluster_label_map.pkl')

    cosine_sim_df = joblib.load('models/cosine_sim_df.pkl')

    return kmeans, scaler, cluster_label_map, cosine_sim_df


@st.cache_data
def load_rfm_segments():

    "Load the pre-computed RFM + segment table for the analytics views."

    try:

        df = pd.read_csv('models/rfm_segments.csv')

        return df

    except FileNotFoundError:

        return None


kmeans, scaler, cluster_label_map, cosine_sim_df = load_models()

rfm_df = load_rfm_segments()


# Pre-rendered icon HTML used across the app (defined once, reused everywhere)
ICON_STOREFRONT = icon("storefront", size = 22)

ICON_TARGET = icon("target", size = 26)

ICON_CART = icon("cart", size = 26)

ICON_CHART = icon("chart", size = 26)

ICON_COMPASS = icon("compass", size = 15)

ICON_SPARKLE = icon("sparkle", size = 15)


# Sidebar Navigation
st.sidebar.markdown(

    f"""
    <div style="padding: 0.4rem 0 1.2rem 0;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; letter-spacing:0.16em;
        color:#2dd4bf; text-transform:uppercase;">customer intelligence</span>
        <h2 style="margin:0.15rem 0 0 0; font-size:1.5rem;">{ICON_STOREFRONT} Shopper Spectrum</h2>
    </div>
    """,

    unsafe_allow_html = True

)


NAV_OPTIONS = ["Home", "Clustering", "Recommendation", "Segment Analytics"]

# Allow the Home page's shortcut buttons to drive navigation: if a hint was set,
# pre-select that page in the radio widget via its session-state key, then clear it.
if st.session_state.get("_nav_hint"):

    st.session_state["_nav_radio"] = st.session_state.pop("_nav_hint")

page = st.sidebar.radio(

    "Navigation",

    options = NAV_OPTIONS,

    label_visibility = "collapsed",

    key = "_nav_radio"

)

st.sidebar.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html = True)

# ---- Light / dark theme toggle (self-contained, defaults to dark) ----
st.sidebar.markdown(
    '<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
    'letter-spacing:0.1em; color:var(--text-2); text-transform:uppercase;">theme</span>',
    unsafe_allow_html = True
)

theme_choice = st.sidebar.radio(

    "Theme",

    options = ["Dark", "Light"],

    index = 0 if st.session_state["theme"] == "dark" else 1,

    horizontal = True,

    label_visibility = "collapsed",

    key = "_theme_radio"

)

new_theme = "dark" if theme_choice == "Dark" else "light"

if new_theme != st.session_state["theme"]:

    st.session_state["theme"] = new_theme

    st.rerun()

st.sidebar.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html = True)

st.sidebar.markdown(

    """
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#5f6b78; line-height:1.9;">
    MODEL &nbsp;&nbsp;&nbsp; KMeans (k=4)<br>
    FEATURES &nbsp; Recency · Frequency · Monetary<br>
    ENGINE &nbsp;&nbsp;&nbsp; Item-based cosine similarity
    </div>
    """,

    unsafe_allow_html = True

)


# Small reusable component: stat card
def stat_card(label, value, delta = None):

    delta_html = f'<span class="delta">{delta}</span>' if delta else ""

    st.markdown(

        f"""
        <div class="stat-card">
            <span class="label">{label}</span>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,

        unsafe_allow_html = True

    )


# HOME
if page == "Home":

    st.markdown('<span class="eyebrow">e-commerce · unsupervised learning</span>', unsafe_allow_html = True)

    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f'{ICON_STOREFRONT} Shopper Spectrum</h1>',
        unsafe_allow_html = True
    )

    st.markdown(

        '<p class="subtext" style="max-width:640px;">Segments customers by purchase behaviour and recommends products from real co-purchase patterns — built on RFM analysis and item-based collaborative filtering.</p>',

        unsafe_allow_html = True

    )

    st.markdown("<br>", unsafe_allow_html = True)

    # ---- top-line stats, computed from the actual saved RFM table ----
    col1, col2, col3, col4 = st.columns(4)

    if rfm_df is not None:

        with col1:

            stat_card("Customers Modelled", f"{len(rfm_df):,}")

        with col2:

            stat_card("Products in Catalogue", f"{len(cosine_sim_df):,}")

        with col3:

            avg_monetary = rfm_df['Monetary'].mean() if 'Monetary' in rfm_df.columns else None

            stat_card("Avg. Customer Spend", f"£{avg_monetary:,.0f}" if avg_monetary else "—")

        with col4:

            stat_card("Segments", "4", delta = "KMeans, silhouette-validated")

    else:

        with col1:

            stat_card("Products in Catalogue", f"{len(cosine_sim_df):,}")

        with col2:

            stat_card("Segments", "4")

        with col3:

            stat_card("Clustering Model", "KMeans")

        with col4:

            stat_card("Recommender", "Cosine similarity")

    st.markdown("<br>", unsafe_allow_html = True)

    st.markdown("---")

    left, right = st.columns([1.1, 1], gap = "large")

    with left:

        st.markdown("### What this does")

        st.markdown(

            """
            <p class="subtext">
            <b style="color:var(--text-0);">Segment customers</b> — every customer is scored on
            Recency, Frequency and Monetary value, scaled, and assigned to one of four behavioural
            groups by a KMeans model trained and validated (silhouette, Davies–Bouldin,
            Calinski–Harabasz) on transaction history.
            </p>
            <p class="subtext">
            <b style="color:var(--text-0);">Recommend products</b> — an item-based collaborative filtering
            engine computes cosine similarity between products from co-purchase patterns, so entering
            a product name surfaces the products most frequently bought alongside it.
            </p>
            """,

            unsafe_allow_html = True

        )

        st.markdown("<br>", unsafe_allow_html = True)

        b1, b2 = st.columns(2)

        with b1:

            if st.button("→ Predict a segment", use_container_width = True):

                st.session_state["_nav_hint"] = "Clustering"

                st.rerun()

        with b2:

            if st.button("→ Get recommendations", use_container_width = True):

                st.session_state["_nav_hint"] = "Recommendation"

                st.rerun()

    with right:

        st.markdown("### Segment reference")

        for name, meta in SEGMENT_INFO.items():

            st.markdown(

                f"""
                <div class="rec-card" style="border-left: 3px solid {meta['color']};">
                    <div>
                        <div class="rec-name">{name}</div>
                        <div style="color:var(--text-2); font-size:0.82rem; margin-top:0.2rem;">{meta['desc']}</div>
                    </div>
                </div>
                """,

                unsafe_allow_html = True

            )


# CLUSTERING — Customer Segmentation Module
elif page == "Clustering":

    st.markdown('<span class="eyebrow">module 01</span>', unsafe_allow_html = True)

    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f'{ICON_TARGET} Customer Segmentation</h1>',
        unsafe_allow_html = True
    )

    st.markdown(

        '<p class="subtext" style="max-width:600px;">Enter a customer\'s RFM profile to predict which behavioural segment they fall into, using the trained KMeans model.</p>',

        unsafe_allow_html = True
    )

    st.markdown("---")

    input_col, chart_col = st.columns([1, 1.15], gap = "large")

    # The KMeans model was trained on RFM values after IQR-winsorization, so predictions
    # for inputs far outside that range are extrapolation, not a validated result.
    # These are the training-data bounds actually seen by the model, pulled from the
    # saved RFM table (models/rfm_segments.csv):
    
    RFM_TRAIN_BOUNDS = {

        "Recency":   (1, 328),

        "Frequency": (1, 11),

        "Monetary":  (3.75, 3691.77)

    }

    with input_col:

        st.markdown("##### Customer profile")

        recency = st.number_input(

            "Recency — days since last purchase",

            min_value = 0,

            max_value = 1000,

            value = 30,

            step = 1

        )

        frequency = st.number_input(

            "Frequency — number of purchases",

            min_value = 1,

            max_value = 1000,

            value = 5,

            step = 1

        )

        monetary = st.number_input(

            "Monetary — total spend (£)",

            min_value = 0.0,

            max_value = 500000.0,

            value = 250.0,

            step = 10.0

        )

        out_of_range = (

            not (RFM_TRAIN_BOUNDS["Recency"][0] <= recency <= RFM_TRAIN_BOUNDS["Recency"][1])

            or not (RFM_TRAIN_BOUNDS["Frequency"][0] <= frequency <= RFM_TRAIN_BOUNDS["Frequency"][1])

            or not (RFM_TRAIN_BOUNDS["Monetary"][0] <= monetary <= RFM_TRAIN_BOUNDS["Monetary"][1])

        )

        if out_of_range:
            
            st.caption(

                "⚠ One or more values fall outside the range the model was trained on "

                f"(Recency {RFM_TRAIN_BOUNDS['Recency'][0]}–{RFM_TRAIN_BOUNDS['Recency'][1]} days, "

                f"Frequency {RFM_TRAIN_BOUNDS['Frequency'][0]}–{RFM_TRAIN_BOUNDS['Frequency'][1]} orders, "

                f"Monetary £{RFM_TRAIN_BOUNDS['Monetary'][0]:,.2f}–£{RFM_TRAIN_BOUNDS['Monetary'][1]:,.2f}). "

                "The prediction below is extrapolated and less reliable — values are capped to the trained range before scoring."

            )

        predict_clicked = st.button("Predict segment", use_container_width = True)

    with chart_col:

        st.markdown("##### Where this customer sits vs. the base")

        if rfm_df is not None and 'Recency' in rfm_df.columns:

            sample = rfm_df.sample(min(1500, len(rfm_df)), random_state = 42)

            fig = px.scatter(

                sample,

                x = "Recency",

                y = "Monetary",

                opacity = 0.35,

                color_discrete_sequence = ["#2dd4bf"]

            )

            fig.add_trace(go.Scatter(

                x = [recency],

                y = [monetary],

                mode = "markers",

                marker = dict(size = 16, color = "#f5a623", symbol = "star", line = dict(width = 1, color = "white")),

                name = "This customer"

            ))

            fig.update_layout(

                paper_bgcolor = "rgba(0,0,0,0)",

                plot_bgcolor = "rgba(0,0,0,0)",

                font_color = "#9aa5b1",

                margin = dict(l = 10, r = 10, t = 10, b = 10),

                height = 340,

                showlegend = False,

                xaxis = dict(gridcolor = "#242b33", title = "Recency (days)"),

                yaxis = dict(gridcolor = "#242b33", title = "Monetary (£)")

            )

            st.plotly_chart(fig, use_container_width = True)

        else:

            st.markdown(

                '<p class="subtext">Upload <code>models/rfm_segments.csv</code> to enable the population comparison chart.</p>',

                unsafe_allow_html = True

            )

    if predict_clicked:

        # Clip to the same IQR-winsorized range the model was trained on (see RFM_TRAIN_BOUNDS
        # above) so predictions are interpolation within the trained space rather than
        # unbounded extrapolation for extreme inputs.
        recency_clipped = min(max(recency, RFM_TRAIN_BOUNDS["Recency"][0]), RFM_TRAIN_BOUNDS["Recency"][1])

        frequency_clipped = min(max(frequency, RFM_TRAIN_BOUNDS["Frequency"][0]), RFM_TRAIN_BOUNDS["Frequency"][1])

        monetary_clipped = min(max(monetary, RFM_TRAIN_BOUNDS["Monetary"][0]), RFM_TRAIN_BOUNDS["Monetary"][1])

        # Arrange input as a DataFrame with the same column names the scaler was fitted on
        input_rfm = pd.DataFrame(

            [[recency_clipped, frequency_clipped, monetary_clipped]],

            columns = ['Recency', 'Frequency', 'Monetary']

        )

        # Scale the input using the same StandardScaler fitted during training
        input_scaled = scaler.transform(input_rfm)

        # Predict the cluster number using the saved KMeans model
        cluster_id = kmeans.predict(input_scaled)[0]

        # Map the cluster number to its business segment label
        segment_label = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")

        meta = SEGMENT_INFO.get(segment_label, {"desc": "", "color": "#2dd4bf"})

        st.markdown(

            f"""
            <div class="segment-banner" style="border-color:{meta['color']};
                background: linear-gradient(135deg, {meta['color']}22, var(--bg-2));">
                <span class="tag" style="color:{meta['color']};">prediction result · cluster {cluster_id}</span>
                <div class="title">{segment_label}</div>
                <div class="desc">{meta['desc']}</div>
            </div>
            """,

            unsafe_allow_html = True

        )

        st.markdown("<br>", unsafe_allow_html = True)

        m1, m2, m3 = st.columns(3)

        with m1:
            stat_card("Recency", f"{recency} days")

        with m2:
            stat_card("Frequency", f"{frequency} orders")

        with m3:
            stat_card("Monetary", f"£{monetary:,.2f}")


# RECOMMENDATION — Product Recommendation Module
elif page == "Recommendation":

    st.markdown('<span class="eyebrow">module 02</span>', unsafe_allow_html = True)

    st.markdown(
        
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        
        f'{ICON_CART} Product Recommender</h1>',
        
        unsafe_allow_html = True
        
    )

    st.markdown(

        '<p class="subtext" style="max-width:600px;">Item-based collaborative filtering — enter a product name to see what customers most often buy alongside it.</p>',

        unsafe_allow_html = True

    )

    st.markdown("---")

    search_col, info_col = st.columns([2, 1], gap = "large")

    with search_col:

        product_input = st.text_input(

            "Product name",

            placeholder = "e.g. WHITE HANGING HEART T-LIGHT HOLDER"

        )

        recommend_clicked = st.button("Find similar products", use_container_width = False)

    with info_col:

        st.markdown(

            f"""
            <div class="stat-card">
                <span class="label">Catalogue size</span>
                <div class="value" style="font-size:1.5rem;">{len(cosine_sim_df):,} SKUs</div>
                <span class="delta">cosine similarity, top-20 cached per item</span>
            </div>
            """,

            unsafe_allow_html = True

        )

    if recommend_clicked:

        product_upper = product_input.upper().strip()

        if not product_upper:

            st.warning("Type a product name to search.")

        else:

            # cosine_sim_df is a dict of {product: pd.Series(top-N similar products)}
            matched = [p for p in cosine_sim_df.keys() if product_upper in p]

            if not matched:

                st.error(f"No product found matching **'{product_input}'**. Check the spelling and try again.")

            else:

                selected_product = matched[0]

                sim_scores = cosine_sim_df[selected_product]

                recommendations = sim_scores.head(5)

                st.markdown("<br>", unsafe_allow_html = True)

                st.markdown(

                    f'<span class="badge">matched product</span> <span style="color:var(--text-0); '

                    f'font-weight:600;">{selected_product}</span>',

                    unsafe_allow_html = True

                )

                st.markdown("<br>", unsafe_allow_html = True)

                st.markdown("##### Frequently bought together")

                max_score = float(recommendations.max()) if len(recommendations) else 1.0

                for rank, (rec_name, score) in enumerate(recommendations.items(), start = 1):

                    pct = int((float(score) / max_score) * 100) if max_score > 0 else 0

                    st.markdown(

                        f"""
                        <div class="rec-card">
                            <div class="rec-rank">#{rank}</div>
                            <div class="rec-name">{rec_name}</div>
                            <div class="rec-score">similarity {float(score):.3f}</div>
                        </div>
                        """,

                        unsafe_allow_html = True

                    )

                if len(matched) > 1:

                    with st.expander(f"{len(matched) - 1} other matching product(s) in catalogue"):

                        for alt in matched[1:11]:

                            st.write(alt)


# SEGMENT ANALYTICS — exploratory view over the saved RFM table
elif page == "Segment Analytics":

    st.markdown('<span class="eyebrow">module 03</span>', unsafe_allow_html = True)

    st.markdown(
        
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        
        f'{ICON_CHART} Segment Analytics</h1>',
        
        unsafe_allow_html = True
        
    )

    st.markdown(

        '<p class="subtext" style="max-width:600px;">A breakdown of the full customer base by segment, computed directly from the saved RFM table.</p>',

        unsafe_allow_html = True

    )

    st.markdown("---")

    if rfm_df is None:

        st.warning(

            "`models/rfm_segments.csv` was not found, so this view has nothing to plot. "

            "Re-run the notebook's save cell to generate it, then push it to the repo."

        )

    else:

        # Figure out which column holds the segment label
        label_col = None

        for candidate in ["Segment", "segment", "KMeans_Cluster", "cluster_label"]:

            if candidate in rfm_df.columns:

                label_col = candidate

                break

        if label_col is None:

            st.warning("Could not find a segment/cluster column in `rfm_segments.csv` to group by.")

        else:

            counts = rfm_df[label_col].value_counts()

            tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Raw data"])

            with tab1:

                c1, c2 = st.columns([1, 1.3], gap = "large")

                with c1:

                    st.markdown("##### Segment share")

                    fig = px.pie(

                        values = counts.values,

                        names = counts.index.astype(str),

                        hole = 0.55,

                        color_discrete_sequence = ["#2dd4bf", "#5b9dd9", "#f5a623", "#ef4b4b", "#7a5cf0"]

                    )

                    fig.update_layout(

                        paper_bgcolor = "rgba(0,0,0,0)",

                        font_color = "#9aa5b1",

                        margin = dict(l = 10, r = 10, t = 10, b = 10),

                        height = 320,

                        legend = dict(orientation = "h", y = -0.15)

                    )

                    st.plotly_chart(fig, use_container_width = True)

                with c2:

                    st.markdown("##### Average RFM by segment")

                    numeric_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in rfm_df.columns]

                    if numeric_cols:

                        profile = rfm_df.groupby(label_col)[numeric_cols].mean().round(1)

                        st.dataframe(profile, use_container_width = True)

                    else:

                        st.write("No Recency/Frequency/Monetary columns found to profile.")

            with tab2:

                numeric_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in rfm_df.columns]

                if numeric_cols:

                    metric_choice = st.selectbox("Metric", numeric_cols)

                    fig = px.box(

                        rfm_df,

                        x = label_col,

                        y = metric_choice,

                        color = label_col,

                        color_discrete_sequence = ["#2dd4bf", "#5b9dd9", "#f5a623", "#ef4b4b", "#7a5cf0"]

                    )

                    fig.update_layout(

                        paper_bgcolor = "rgba(0,0,0,0)",

                        plot_bgcolor = "rgba(0,0,0,0)",

                        font_color = "#9aa5b1",

                        margin = dict(l = 10, r = 10, t = 10, b = 10),

                        height = 420,

                        showlegend = False,

                        xaxis = dict(gridcolor = "#242b33"),

                        yaxis = dict(gridcolor = "#242b33")

                    )

                    st.plotly_chart(fig, use_container_width = True)

                else:

                    st.write("No numeric RFM columns found to plot.")

            with tab3:

                st.markdown("##### Full table")

                st.dataframe(rfm_df, use_container_width = True, height = 420)

                csv_bytes = rfm_df.to_csv(index = False).encode("utf-8")

                st.download_button(

                    "Download as CSV",

                    data = csv_bytes,

                    file_name = "rfm_segments_export.csv",

                    mime = "text/csv"

                )


# Footer
st.markdown(

    '<div class="footer-note">SHOPPER SPECTRUM · KMeans clustering + item-based collaborative filtering</div>',

    unsafe_allow_html = True
)