import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --- Session state defaults ---
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# --- Elegant font import ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Poppins:wght@300;400;500&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: 0.5px; }
    </style>
""", unsafe_allow_html=True)

# --- Top bar: appearance selector on the right ---
top_left, top_right = st.columns([5, 1])
with top_right:
    theme_choice = st.selectbox(
        "Appearance",
        ["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        label_visibility="collapsed"
    )
    st.session_state.theme = "Dark" if "Dark" in theme_choice else "Light"

IS_DARK = st.session_state.theme == "Dark"

# --- Elegant color palette used consistently across every chart ---
PALETTE = ["#B08968", "#7D8570", "#9A8C98", "#C9A987", "#4A4E69", "#A5A58D"]
ACCENT = "#B08968"
POSITIVE = "#7D8570"
NEGATIVE = "#B5654B"

TEXT_COLOR = "#f2f2f2" if IS_DARK else "#2b2b2b"
BG_COLOR = "#12141a" if IS_DARK else "#ffffff"
GRID_COLOR = "#2c2f38" if IS_DARK else "#e6e2dc"
CARD_BG = "#1c2129" if IS_DARK else "#f7f4f0"

def style_fig(fig, ax_list=None):
    """Apply a consistent, elegant, theme-aware style to any matplotlib figure."""
    fig.patch.set_facecolor(BG_COLOR)
    axes = ax_list if ax_list else fig.get_axes()
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

# --- Theme CSS: page background, widget labels, metrics, dropdowns, and NATIVE TABS ---
# Everything here lives in the SAME document as the rest of the app (no iframe),
# so it reliably follows Light/Dark in every browser.
st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    [data-testid="stMarkdownContainer"], label, p, span, h1, h2, h3, h4 {{ color: {TEXT_COLOR} !important; }}
    .stMetric, div[data-testid="stMetric"] {{ background-color: {CARD_BG}; border-radius: 8px; padding: 10px; }}

    /* Dropdowns (selectbox) */
    div[data-baseweb="select"] > div {{ background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important; border-color: {GRID_COLOR} !important; }}
    div[data-baseweb="popover"] div {{ background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important; }}
    li[role="option"] {{ background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important; }}

    /* Number inputs */
    div[data-testid="stNumberInput"] input {{ background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important; }}

    /* --- Native tabs styled as an elegant horizontal navbar --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 36px;
        border-bottom: 1px solid {GRID_COLOR};
        justify-content: center;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        background-color: transparent;
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
        font-weight: 400;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {TEXT_COLOR};
        border-bottom: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT_COLOR} !important;
        border-bottom: 2px solid {TEXT_COLOR} !important;
        font-weight: 500;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent; }}
    </style>
""", unsafe_allow_html=True)

# --- Load saved model artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("../models/final_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    cols_to_scale = joblib.load("../models/cols_to_scale.pkl")
    feature_columns = joblib.load("../models/feature_columns.pkl")
    all_models = joblib.load("../models/all_models.pkl")
    return model, scaler, cols_to_scale, feature_columns, all_models

model, scaler, cols_to_scale, feature_columns, all_models = load_artifacts()

# --- Load raw data once, for the Data Insights page ---
@st.cache_data
def load_raw_data():
    return pd.read_csv("../data/train.csv")

raw_df = load_raw_data()

st.markdown(f"<h1 style='text-align:center; margin-top: 10px; color:{TEXT_COLOR};'>🏠 House Price Prediction System</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{'#bbb' if IS_DARK else '#777'};'>Powered by a tuned Gradient Boosting model trained on real Ames, Iowa housing data.</p>", unsafe_allow_html=True)
st.divider()

# --- Native tabs: same document as everything else, so theming always applies ---
tab_predict, tab_insights, tab_performance = st.tabs(["🔮  PREDICT PRICE", "📊  DATA INSIGHTS", "📈  MODEL PERFORMANCE"])

# =====================================================================
# TAB 1: PREDICT PRICE
# =====================================================================
with tab_predict:

    st.header("📋 Enter House Details")

    selected_model_name = st.selectbox(
        "🤖 Choose which model makes the prediction",
        ["Gradient Boosting (Tuned - Best)", "Random Forest", "Ridge Regression", "Linear Regression"]
    )
    model_key_map = {
        "Gradient Boosting (Tuned - Best)": "Gradient Boosting",
        "Random Forest": "Random Forest",
        "Ridge Regression": "Ridge Regression",
        "Linear Regression": "Linear Regression",
    }
    active_model = model if selected_model_name == "Gradient Boosting (Tuned - Best)" else all_models[model_key_map[selected_model_name]]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Size")
        gr_liv_area = st.number_input("Above-ground living area (sq ft)", min_value=200, max_value=6000, value=1500)
        total_bsmt_sf = st.number_input("Total basement area (sq ft)", min_value=0, max_value=3000, value=800)
        lot_area = st.number_input("Lot area (sq ft)", min_value=500, max_value=50000, value=9000)
        garage_area = st.number_input("Garage area (sq ft)", min_value=0, max_value=1500, value=400)

    with col2:
        st.subheader("Rooms & Quality")
        overall_qual = st.slider("Overall material & finish quality (1=Poor, 10=Excellent)", 1, 10, 6)
        full_bath = st.number_input("Full bathrooms", min_value=0, max_value=5, value=2)
        half_bath = st.number_input("Half bathrooms", min_value=0, max_value=3, value=0)
        bedroom = st.number_input("Bedrooms above ground", min_value=0, max_value=10, value=3)
        kitchen_qual = st.selectbox("Kitchen quality", ["Excellent", "Good", "Average", "Fair", "Poor"])

    with col3:
        st.subheader("Location & Age")
        neighborhood = st.selectbox("Neighborhood", sorted([
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert",
            "NridgHt", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
            "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
            "SWISU", "Blmngtn", "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste"
        ]))
        year_built = st.number_input("Year built", min_value=1870, max_value=2025, value=1990)
        year_sold = st.number_input("Year sold", min_value=1870, max_value=2025, value=2010)
        has_fireplace = st.checkbox("Has a fireplace", value=True)
        has_pool = st.checkbox("Has a pool", value=False)

    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_button:
        defaults = joblib.load("../models/feature_defaults.pkl")
        input_row = defaults.copy()

        input_row["GrLivArea"] = gr_liv_area
        input_row["TotalBsmtSF"] = total_bsmt_sf
        input_row["LotArea"] = lot_area
        input_row["GarageArea"] = garage_area
        input_row["OverallQual"] = overall_qual
        input_row["FullBath"] = full_bath
        input_row["HalfBath"] = half_bath
        input_row["BedroomAbvGr"] = bedroom
        input_row["YearBuilt"] = year_built
        input_row["YrSold"] = year_sold

        kitchen_map = {"Excellent": 5, "Good": 4, "Average": 3, "Fair": 2, "Poor": 1}
        input_row["KitchenQual"] = kitchen_map[kitchen_qual]

        input_row["HasFireplace"] = int(has_fireplace)
        input_row["HasPool"] = int(has_pool)

        input_row["TotalSF"] = input_row["TotalBsmtSF"] + input_row["1stFlrSF"] + input_row["2ndFlrSF"]
        input_row["HouseAge"] = input_row["YrSold"] - input_row["YearBuilt"]
        input_row["TotalBath"] = (input_row["FullBath"] + 0.5 * input_row["HalfBath"] +
                                   input_row["BsmtFullBath"] + 0.5 * input_row["BsmtHalfBath"])
        input_row["HasGarage"] = int(input_row["GarageArea"] > 0)
        input_row["HasBasement"] = int(input_row["TotalBsmtSF"] > 0)

        for col in feature_columns:
            if col.startswith("Neighborhood_"):
                input_row[col] = 0
        neighborhood_col = f"Neighborhood_{neighborhood}"
        if neighborhood_col in feature_columns:
            input_row[neighborhood_col] = 1

        input_df = pd.DataFrame([input_row])[feature_columns]
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        prediction = active_model.predict(input_df)[0]

        model_rmse = 19966
        lower_bound = prediction - model_rmse
        upper_bound = prediction + model_rmse

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Lower Estimate", f"${lower_bound:,.0f}")
        with col_b:
            st.metric("Predicted Price", f"${prediction:,.0f}")
        with col_c:
            st.metric("Upper Estimate", f"${upper_bound:,.0f}")

        st.caption(f"This range reflects the model's typical prediction error (±${model_rmse:,.0f} RMSE on test data), "
                   f"not a statistical confidence interval.")

        st.divider()
        st.subheader("📊 What Influenced This Prediction Most")

        if hasattr(active_model, "feature_importances_"):
            importances = active_model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color=ACCENT)
            ax.set_xlabel("Importance")
            ax.set_title(f"Top 10 Most Important Features ({selected_model_name})")
            style_fig(fig, [ax])
            st.pyplot(fig)

            st.caption("These are the features the model relies on most heavily across all predictions, "
                       "not specific to this single house.")
        elif hasattr(active_model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": feature_columns,
                "Coefficient": active_model.coef_
            })
            coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
            coef_df = coef_df.sort_values("AbsCoefficient", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = [POSITIVE if c > 0 else NEGATIVE for c in coef_df["Coefficient"][::-1]]
            ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=colors)
            ax.set_xlabel("Coefficient (sage = increases price, terracotta = decreases price)")
            ax.set_title(f"Top 10 Influential Features ({selected_model_name})")
            style_fig(fig, [ax])
            st.pyplot(fig)

            st.caption("Linear models show coefficients instead of tree-based feature importance — "
                       "a positive coefficient increases predicted price, negative decreases it.")

# =====================================================================
# TAB 2: DATA INSIGHTS
# =====================================================================
with tab_insights:

    st.header("📊 Data Insights")
    st.markdown("Exploring patterns in the training data used to build this model.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sale Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(raw_df["SalePrice"], bins=40, color=ACCENT, edgecolor=BG_COLOR)
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Number of Houses")
        style_fig(fig, [ax])
        st.pyplot(fig)

    with col2:
        st.subheader("Price vs. Living Area")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(raw_df["GrLivArea"], raw_df["SalePrice"], alpha=0.5, color=POSITIVE, edgecolors="none", s=25)
        ax.set_xlabel("Above-Ground Living Area (sq ft)")
        ax.set_ylabel("Sale Price ($)")
        style_fig(fig, [ax])
        st.pyplot(fig)

    st.subheader("Average Price by Neighborhood (Top 10)")
    avg_price = raw_df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(avg_price.index, avg_price.values, color=PALETTE[:len(avg_price)] if len(avg_price) <= len(PALETTE) else ACCENT)
    ax.set_ylabel("Average Sale Price ($)")
    ax.tick_params(axis='x', rotation=30)
    style_fig(fig, [ax])
    st.pyplot(fig)

    st.subheader("Correlation with Sale Price (Top Numeric Features)")
    numeric_df = raw_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["SalePrice"].sort_values(ascending=False).drop("SalePrice").head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index[::-1], corr.values[::-1], color="#4A4E69")
    ax.set_xlabel("Correlation with Sale Price")
    style_fig(fig, [ax])
    st.pyplot(fig)

# =====================================================================
# TAB 3: MODEL PERFORMANCE
# =====================================================================
with tab_performance:

    st.header("📈 Model Performance")
    st.markdown("Comparison of all models trained during development.")

    results_data = {
        "Model": ["Gradient Boosting", "Random Forest", "Ridge Regression", "Linear Regression"],
        "MAE": [14571, 16134, 18812, 19588],
        "RMSE": [20418, 23910, 25530, 27460],
        "R2 Score": [0.9245, 0.8965, 0.8820, 0.8635],
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results_df["Model"], results_df["RMSE"], color=ACCENT)
        ax.set_title("RMSE by Model (lower is better)")
        ax.tick_params(axis='x', rotation=30)
        style_fig(fig, [ax])
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results_df["Model"], results_df["R2 Score"], color=POSITIVE)
        ax.set_title("R² Score by Model (higher is better)")
        ax.tick_params(axis='x', rotation=30)
        style_fig(fig, [ax])
        st.pyplot(fig)

    st.divider()
    st.subheader("Before vs. After Hyperparameter Tuning (Gradient Boosting)")
    tuning_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2 Score"],
        "Before Tuning": [14571, 20418, 0.9245],
        "After Tuning": [13887, 19966, 0.9273],
    })
    st.dataframe(tuning_df, use_container_width=True)
    st.caption("RandomizedSearchCV (30 combinations, 5-fold cross-validation) improved every metric.")