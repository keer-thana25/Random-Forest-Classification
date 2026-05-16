# ================================
# RANDOM FOREST CLASSIFICATION APP
# General-Purpose Dynamic Version
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import plot_tree

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Random Forest Studio",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --forest-green:   #2d6a4f;
    --leaf-green:     #40916c;
    --mint:           #74c69d;
    --accent:         #52b788;
    --warning:        #e07a5f;
    --radius-card:    14px;
    --radius-sm:      8px;
    --shadow-card:    0 2px 16px rgba(0,0,0,0.08);
    --shadow-hover:   0 6px 24px rgba(0,0,0,0.13);
    --transition:     0.22s ease;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌲🌳🌲🌳🌲";
    position: absolute; right: 32px; top: 50%;
    transform: translateY(-50%);
    font-size: 2.8rem; letter-spacing: 6px; opacity: 0.25;
}
.hero h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero p  { color: #b7e4c7; font-size: 1.05rem; margin: 0; font-weight: 400; }

.section-heading {
    font-size: 1.25rem; font-weight: 700;
    margin: 28px 0 14px 0; padding-left: 12px;
    border-left: 4px solid #40916c;
    color: inherit; letter-spacing: -0.3px;
}

.card {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-card);
    padding: 22px 24px; margin-bottom: 18px;
    box-shadow: var(--shadow-card);
    transition: box-shadow var(--transition);
}
.card:hover { box-shadow: var(--shadow-hover); }

.stat-chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--chip-bg, #f0faf5);
    border: 1px solid #b7e4c7; border-radius: 50px;
    padding: 6px 16px; font-size: 0.88rem; font-weight: 600;
    color: #1b4332; margin: 4px;
}

.concept-badge {
    display: inline-flex; align-items: flex-start; gap: 10px;
    background: var(--badge-bg, #f6fff9);
    border: 1px solid #b7e4c7; border-radius: var(--radius-card);
    padding: 14px 18px; margin-bottom: 12px;
    width: 100%; box-sizing: border-box;
}
.concept-badge .icon { font-size: 1.4rem; flex-shrink: 0; }
.concept-badge .text b  { display: block; font-size: 0.92rem; font-weight: 700; margin-bottom: 3px; color: #1b4332; }
.concept-badge .text span { font-size: 0.85rem; color: var(--body-muted, #555); line-height: 1.5; }

.adv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.adv-item {
    display: flex; align-items: center; gap: 9px;
    background: var(--chip-bg, #f0faf5); border-radius: var(--radius-sm);
    padding: 11px 14px; font-size: 0.88rem; font-weight: 500;
    color: var(--body-color, #222); border: 1px solid #d8f3dc;
}

.step-row { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
.step-num {
    background: #40916c; color: #fff; border-radius: 50%;
    min-width: 32px; height: 32px; display: flex; align-items: center;
    justify-content: center; font-size: 0.85rem; font-weight: 700;
    flex-shrink: 0; margin-top: 2px;
}
.step-body b    { font-weight: 700; font-size: 0.93rem; display: block; margin-bottom: 3px; color: var(--body-color, #222); }
.step-body span { font-size: 0.84rem; color: var(--body-muted, #555); line-height: 1.5; }

.analogy-box {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-radius: var(--radius-card); padding: 18px 22px;
    margin: 16px 0; border-left: 4px solid #40916c;
}
.analogy-box p { margin: 0; color: #1b4332; font-size: 0.92rem; line-height: 1.6; }
.analogy-box b { font-weight: 700; }

.result-survived {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border: 2px solid #52b788; border-radius: var(--radius-card);
    padding: 24px 28px; text-align: center;
}
.result-survived h2 { color: #1b4332; margin: 0 0 6px 0; font-size: 1.5rem; }
.result-survived p  { color: #2d6a4f; margin: 0; font-size: 1rem; }

.result-not-survived {
    background: linear-gradient(135deg, #fde8e4, #f9c8bf);
    border: 2px solid #e07a5f; border-radius: var(--radius-card);
    padding: 24px 28px; text-align: center;
}
.result-not-survived h2 { color: #7a2c1a; margin: 0 0 6px 0; font-size: 1.5rem; }
.result-not-survived p  { color: #9e3b23; margin: 0; font-size: 1rem; }

.prob-bar-wrap {
    background: rgba(0,0,0,0.08); border-radius: 50px;
    height: 10px; margin: 12px 0 4px 0; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #52b788, #2d6a4f);
    transition: width 0.5s ease;
}

.depth-info {
    background: var(--chip-bg, #f0faf5); border: 1px solid #b7e4c7;
    border-radius: var(--radius-sm); padding: 12px 16px;
    font-size: 0.84rem; color: var(--body-muted, #444);
    margin-top: 10px; line-height: 1.55;
}
.depth-info b { color: #1b4332; }

.col-info-card {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-sm); padding: 13px 15px; margin-bottom: 10px;
}
.col-info-card .col-name { font-weight: 700; font-size: 0.88rem; color: #2d6a4f; margin-bottom: 3px; }
.col-info-card .col-desc { font-size: 0.82rem; color: var(--body-muted, #555); line-height: 1.45; }

.warning-box {
    background: linear-gradient(135deg, #fde8e4, #f9c8bf);
    border: 2px solid #e07a5f; border-radius: var(--radius-card);
    padding: 20px 24px; margin: 16px 0;
}
.warning-box h3 { color: #7a2c1a; margin: 0 0 8px 0; }
.warning-box p  { color: #9e3b23; margin: 0; font-size: 0.92rem; line-height: 1.6; }

@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1e2a23; --card-border: rgba(255,255,255,0.08);
        --chip-bg: #1b3028; --badge-bg: #1b3028;
        --body-color: #e8f5ed; --body-muted: #9dbfab;
    }
    .hero h1 { color: #fff; }
    .hero p  { color: #b7e4c7; }
    .section-heading { color: #b7e4c7; }
    .concept-badge .text b { color: #95d5b2; }
    .analogy-box { background: linear-gradient(135deg, #1b3828, #1e4030); }
    .analogy-box p { color: #b7e4c7; }
    .adv-item { background: #1b3028; border-color: #2d5a42; color: #cde9d9; }
    .stat-chip { background: #1b3028; color: #95d5b2; }
    .step-body b { color: #d8f3dc; }
    .step-body span { color: #9dbfab; }
    .col-info-card { background: #1e2a23; border-color: rgba(255,255,255,0.07); }
    .col-info-card .col-desc { color: #9dbfab; }
    .depth-info { background: #1b3028; color: #9dbfab; }
    .result-survived { background: linear-gradient(135deg, #1b3828, #1e4030); border-color: #52b788; }
    .result-survived h2 { color: #95d5b2; }
    .result-survived p  { color: #74c69d; }
    .result-not-survived { background: linear-gradient(135deg, #3a1f1a, #4a2620); border-color: #e07a5f; }
    .result-not-survived h2 { color: #f4b3a4; }
    .result-not-survived p  { color: #e8998c; }
    .warning-box { background: linear-gradient(135deg, #3a1f1a, #4a2620); border-color: #e07a5f; }
    .warning-box h3 { color: #f4b3a4; }
    .warning-box p  { color: #e8998c; }
}

.stSlider label, .stSelectbox label, .stNumberInput label { font-weight: 600; font-size: 0.88rem; }

.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #40916c);
    color: #fff; border: none; border-radius: var(--radius-sm);
    padding: 10px 28px; font-size: 0.95rem; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer;
    transition: opacity var(--transition), transform var(--transition);
    box-shadow: 0 3px 10px rgba(45,106,79,0.3);
}
.stButton > button:hover {
    opacity: 0.88; transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(45,106,79,0.35);
}

[data-testid="stMetric"] {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-card); padding: 16px 20px;
    box-shadow: var(--shadow-card);
}
[data-testid="stSidebar"] { border-right: 1px solid var(--card-border, rgba(0,0,0,0.06)); }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; font-weight: 500; padding: 5px 0; }
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ================================
# HELPER FUNCTIONS
# ================================

def apply_plot_theme(fig, ax_list=None):
    """Neutral theme readable in both light & dark mode."""
    fig.patch.set_facecolor("none")
    for ax in (ax_list or fig.axes):
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_color("#80f0a7")
        ax.tick_params(colors="#6c9e82", labelsize=9)
        ax.xaxis.label.set_color("#6c9e82")
        ax.yaxis.label.set_color("#6c9e82")
        ax.title.set_color("#2d6a4f")


def load_titanic():
    """Load and normalise the built-in Titanic dataset."""
    df = sns.load_dataset("titanic")
    rename_map = {
        "survived": "Survived", "pclass": "Pclass", "sex": "Sex",
        "age": "Age", "sibsp": "SibSp", "parch": "Parch",
        "fare": "Fare", "embarked": "Embarked",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df


def detect_target_column(df):
    """
    Auto-detect the most likely target column for classification.
    Priority: known names → low-cardinality categoricals → binary columns.
    Returns column name or None.
    """
    priority_names = [
        "target", "label", "class", "outcome", "result",
        "survived", "species", "diagnosis", "status", "category",
        "Survived", "Species", "Diagnosis", "Status", "Category",
    ]
    for name in priority_names:
        if name in df.columns:
            return name

    # Fallback: find best column by cardinality heuristics
    best = None
    best_score = -1
    n = len(df)
    for col in df.columns:
        nuniq = df[col].nunique()
        if nuniq < 2 or nuniq > 20:
            continue
        # Penalise ID-like columns (high unique ratio)
        unique_ratio = nuniq / n
        if unique_ratio > 0.5:
            continue
        # Score: prefer low cardinality, non-numeric first
        is_cat = df[col].dtype == object or str(df[col].dtype) == "category"
        score = (1 / (nuniq + 1)) + (0.5 if is_cat else 0)
        if score > best_score:
            best_score = score
            best = col
    return best


def validate_classification_dataset(df, target_col):
    """
    Returns (is_valid: bool, reason: str).
    Checks if dataset is suitable for RF classification.
    """
    if target_col is None:
        return False, "No suitable target column found."

    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found."

    target = df[target_col].dropna()
    n = len(df)
    nuniq = target.nunique()

    if n < 30:
        return False, f"Too few rows ({n}). Need at least 30 samples."

    if nuniq < 2:
        return False, "Target column has fewer than 2 unique values."

    if nuniq > 50:
        return False, (
            f"Target column has {nuniq} unique values — this looks like a regression "
            "target, not a classification target."
        )

    # Check if target is continuous numeric (regression)
    if pd.api.types.is_numeric_dtype(target):
        unique_ratio = nuniq / n
        if unique_ratio > 0.05 and nuniq > 20:
            return False, (
                f"Target column '{target_col}' appears to be continuous numerical "
                f"({nuniq} unique values out of {n} rows). This is a regression dataset."
            )

    # Check minimum samples per class
    min_class_count = target.value_counts().min()
    if min_class_count < 5:
        return False, (
            f"Some classes have fewer than 5 samples (min={min_class_count}). "
            "Need more samples per class."
        )

    # Check at least 2 non-target columns usable as features
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2:
        return False, "Not enough feature columns (need at least 2 besides target)."

    return True, "OK"


def preprocess_dataset(df, target_col):
    """
    Full preprocessing pipeline for any dataset:
    - Drop ID/useless columns
    - Fill missing values
    - Encode categoricals
    - Return X, y, scaler, feature_names
    """
    df = df.copy()

    # --- Drop high-cardinality / ID-like columns ---
    id_keywords = [
        "id", "name", "ticket", "cabin", "index", "passport",
        "passengerid", "passenger_id", "email", "phone", "address",
    ]
    n = len(df)
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        # Drop if name matches ID keywords
        if any(kw in col_lower for kw in id_keywords):
            drop_cols.append(col)
            continue
        # Drop if nearly all values are unique (ID-like)
        if df[col].dtype == object and df[col].nunique() / n > 0.8:
            drop_cols.append(col)

    # Also drop Titanic-specific helper cols if present
    titanic_drops = ["who", "alive", "adult_male", "embark_town", "class", "alone", "deck"]
    for col in titanic_drops:
        if col in df.columns and col not in drop_cols:
            drop_cols.append(col)

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Fill missing values ---
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # --- Encode categoricals ---
    df = pd.get_dummies(df, drop_first=True)

    # --- Split features / target ---
    # After get_dummies target_col might be bool-encoded; handle gracefully
    # Find the target column (may have been encoded if it was categorical with >2 classes)
    # Strategy: re-encode target separately before get_dummies
    return df, drop_cols


def prepare_model_data(df_raw, target_col):
    """
    Full pipeline: preprocess → split → scale.
    Returns X_train, X_test, y_train, y_test, scaler, X (DataFrame), y (Series), drop_cols.
    """
    df = df_raw.copy()

    # Separate target before encoding to keep it clean
    y_raw = df[target_col].copy()
    df_features = df.drop(columns=[target_col])

    # Drop IDs / useless cols
    id_keywords = ["id", "name", "ticket", "cabin", "index", "passport",
                   "passengerid", "passenger_id", "email", "phone", "address"]
    titanic_drops = ["who", "alive", "adult_male", "embark_town", "class", "alone", "deck"]
    n = len(df)
    drop_cols = []
    for col in df_features.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_keywords):
            drop_cols.append(col)
            continue
        if any(col == td for td in titanic_drops):
            drop_cols.append(col)
            continue
        if df_features[col].dtype == object and df_features[col].nunique() / n > 0.8:
            drop_cols.append(col)

    df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], inplace=True)

    # Fill missing values
    for col in df_features.columns:
        if df_features[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df_features[col]):
            df_features[col] = df_features[col].fillna(df_features[col].median())
        else:
            mode_val = df_features[col].mode()
            df_features[col] = df_features[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # Encode categoricals
    df_features = pd.get_dummies(df_features, drop_first=True)

    # Encode target (label encode if needed)
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        y = pd.Categorical(y_raw).codes
        class_names = list(pd.Categorical(y_raw).categories)
    else:
        y = y_raw.reset_index(drop=True)
        class_names = sorted(y.unique().tolist())

    X = df_features.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None
    )

    return X_train, X_test, y_train, y_test, scaler, X, y, drop_cols, class_names


def remove_outliers(df, target_col):
    """IQR-based outlier removal on numeric columns except target."""
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col == target_col:
            continue
        Q1, Q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        df_out = df_out[
            (df_out[col] >= Q1 - 1.5 * IQR) &
            (df_out[col] <= Q3 + 1.5 * IQR)
        ]
    return df_out


def build_prediction_form(df_raw, target_col, X_columns, scaler):
    """
    Dynamically generate prediction form inputs from dataset features.
    Returns input_scaled array ready for model.predict().
    """
    # Use original columns (before get_dummies) for the form
    feature_cols_raw = [c for c in df_raw.columns if c != target_col]

    # Filter out ID/useless
    id_keywords = ["id", "name", "ticket", "cabin", "index", "passport",
                   "passengerid", "passenger_id", "email", "phone", "address"]
    titanic_drops = ["who", "alive", "adult_male", "embark_town", "class", "alone", "deck"]
    n = len(df_raw)
    form_cols = []
    for col in feature_cols_raw:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_keywords):
            continue
        if col in titanic_drops:
            continue
        if df_raw[col].dtype == object and df_raw[col].nunique() / n > 0.8:
            continue
        form_cols.append(col)

    if not form_cols:
        st.warning("No suitable feature columns found for prediction form.")
        return None

    # Split into groups of 3 for columns
    num_cols_ui = min(3, len(form_cols))
    cols_ui = st.columns(num_cols_ui)

    user_input = {}
    for i, col in enumerate(form_cols):
        col_data = df_raw[col].dropna()
        with cols_ui[i % num_cols_ui]:
            if pd.api.types.is_numeric_dtype(col_data):
                col_min = float(col_data.min())
                col_max = float(col_data.max())
                col_med = float(col_data.median())
                if col_data.nunique() <= 10 and col_data.nunique() > 1:
                    # Treat as ordinal slider
                    unique_vals = sorted(col_data.unique().tolist())
                    val = st.select_slider(f"**{col}**", options=unique_vals, value=unique_vals[len(unique_vals)//2])
                    user_input[col] = val
                else:
                    val = st.slider(f"**{col}**", min_value=col_min, max_value=col_max, value=col_med)
                    user_input[col] = val
            else:
                unique_vals = col_data.unique().tolist()
                val = st.selectbox(f"**{col}**", unique_vals)
                user_input[col] = val

    # Build a single-row DataFrame matching the training features
    input_row = pd.DataFrame([user_input])

    # Fill any missing numeric cols with median
    for col in input_row.columns:
        if input_row[col].isnull().any():
            input_row[col] = df_raw[col].median() if pd.api.types.is_numeric_dtype(df_raw[col]) else df_raw[col].mode()[0]

    # One-hot encode matching training schema
    input_encoded = pd.get_dummies(input_row, drop_first=True)
    input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    return input_scaled


def show_invalid_dataset_message(reason):
    """Show a clean error UI for invalid datasets."""
    st.markdown(f"""
    <div class="warning-box">
        <h3>❌ This dataset is not suitable for Random Forest Classification</h3>
        <p><strong>Reason:</strong> {reason}</p>
        <p style="margin-top:10px;">Please upload a classification dataset suitable for Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">💡 Example Datasets That Work Well</div>', unsafe_allow_html=True)

    examples = [
        ("🌸", "Iris Dataset", "3-class flower species classification — sepal/petal features."),
        ("🎗️", "Breast Cancer Dataset", "Binary classification: malignant vs benign tumors."),
        ("🍷", "Wine Dataset", "3-class wine origin classification from chemical features."),
        ("❤️", "Heart Disease Dataset", "Binary: presence or absence of heart disease."),
        ("🚢", "Titanic Dataset", "Binary survival prediction — the built-in default."),
    ]
    cols3 = st.columns(3)
    for i, (icon, name, desc) in enumerate(examples):
        with cols3[i % 3]:
            st.markdown(f"""
            <div class="col-info-card">
                <div class="col-name">{icon} {name}</div>
                <div class="col-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.info("Remove the uploaded file in the sidebar to return to the default Titanic dataset, or upload a different classification CSV.")


# ================================
# HERO HEADER
# ================================

st.markdown("""
<div class="hero">
    <h1>🌲 Random Forest Classification Studio</h1>
    <p>General-Purpose ML Classification · Titanic Default · Upload Any Dataset</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================

with st.sidebar:
    st.markdown("### 📌 Navigation")
    section = st.radio(
        "Go to section",
        ("🌳 What is Random Forest?", "🤖 Random Forest Classification"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown(
        "<small style='color:#6c9e82;'>Built with Streamlit · Scikit-learn</small>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════
# SECTION 1 — WHAT IS RANDOM FOREST?
# ════════════════════════════════════════════════

if section == "🌳 What is Random Forest?":

    st.markdown('<div class="section-heading">📖 What is Random Forest?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p style="font-size:0.97rem; line-height:1.75; margin:0;">
            A <strong>Random Forest</strong> is a machine learning algorithm that builds
            <strong>many decision trees</strong> and combines their predictions to give a final answer.
            Instead of trusting a single tree (which can make mistakes), it asks hundreds of trees
            and picks the answer that the <strong>majority vote</strong> for — just like asking
            a crowd of experts rather than one person.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analogy-box">
        <p>
        🎓 <b>Real-life analogy:</b> Imagine you want to decide which movie to watch tonight.
        Instead of asking only one friend (who might have bad taste), you ask
        <b>50 friends</b>. Whatever movie most of them recommend — that's what you watch.
        Random Forest works exactly the same way with decision trees.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🧩 Core Concepts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-badge"><div class="icon">🌲</div><div class="text">
        <b>Decision Tree</b>
        <span>A single model that splits data into branches based on yes/no questions, like a flowchart.</span>
        </div></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-badge"><div class="icon">🗳️</div><div class="text">
        <b>Majority Voting</b>
        <span>Each tree votes. The class with the most votes becomes the final prediction.</span>
        </div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-badge"><div class="icon">🎲</div><div class="text">
        <b>Random Sampling</b>
        <span>Each tree trains on a random subset of the data, so every tree is slightly different.</span>
        </div></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-badge"><div class="icon">🏡</div><div class="text">
        <b>The Forest</b>
        <span>Many diverse trees together. Diversity is the key — it reduces errors from any single tree.</span>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">⚙️ How It Works — Step by Step</div>', unsafe_allow_html=True)

    steps = [
        ("Collect Data", "Feed the model your dataset — features like age, size, or category columns."),
        ("Build Many Trees", "The algorithm creates hundreds of decision trees, each on a random data sample."),
        ("Each Tree Predicts", "For a new data point, every individual tree makes its own prediction independently."),
        ("Vote & Decide", "All trees cast their vote. The majority class wins and becomes the final prediction."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{i}</div>
            <div class="step-body"><b>{title}</b><span>{desc}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">✅ Why Use Random Forest?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="adv-grid">
        <div class="adv-item">🎯 <span><b>High Accuracy</b> — Better than a single tree</span></div>
        <div class="adv-item">🛡️ <span><b>Avoids Overfitting</b> — Generalises well</span></div>
        <div class="adv-item">📦 <span><b>Handles Large Data</b> — Scales easily</span></div>
        <div class="adv-item">🔍 <span><b>Feature Importance</b> — Tells you what matters</span></div>
        <div class="adv-item">🧩 <span><b>Missing Values</b> — Tolerates imperfect data</span></div>
        <div class="adv-item">⚡ <span><b>Fast to Train</b> — Trees train in parallel</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🎨 Visual: How Random Forest Decides</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    inp = FancyBboxPatch((0.1, 1.7), 1.6, 1.6, boxstyle="round,pad=0.1",
                         facecolor="#d8f3dc", edgecolor="#40916c", linewidth=2)
    ax.add_patch(inp)
    ax.text(0.9, 2.5, "📊\nInput\nData", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1b4332")

    tree_colors = ["#b7e4c7", "#95d5b2", "#74c69d"]
    tree_labels = ["Tree 1", "Tree 2", "Tree 3"]
    tree_votes  = ["Class A ✓", "Class B ✗", "Class A ✓"]
    tree_x = [2.9, 3.9, 4.9]
    for tx, tc, tl, tv in zip(tree_x, tree_colors, tree_labels, tree_votes):
        box = FancyBboxPatch((tx, 1.2), 1.4, 2.6, boxstyle="round,pad=0.1",
                             facecolor=tc, edgecolor="#40916c", linewidth=1.5)
        ax.add_patch(box)
        ax.text(tx+0.7, 3.45, f"🌲 {tl}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1b4332")
        ax.text(tx+0.7, 2.55, "?", ha="center", va="center",
                fontsize=28, color="#2d6a4f", alpha=0.35)
        ax.text(tx+0.7, 1.58, tv, ha="center", va="center",
                fontsize=7.5, color="#1b4332", fontweight="bold")
        ax.annotate("", xy=(tx+0.05, 2.5), xytext=(1.7, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#40916c", lw=1.5))

    vbox = FancyBboxPatch((6.8, 1.5), 1.6, 2.0, boxstyle="round,pad=0.1",
                          facecolor="#52b788", edgecolor="#2d6a4f", linewidth=2)
    ax.add_patch(vbox)
    ax.text(7.6, 2.8, "🗳️ Vote", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#fff")
    ax.text(7.6, 2.3, "2 vs 1", ha="center", va="center",
            fontsize=8.5, color="#d8f3dc")

    for tx in tree_x:
        ax.annotate("", xy=(6.8, 2.5), xytext=(tx+1.45, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#40916c", lw=1.5))

    fbox = FancyBboxPatch((8.6, 1.7), 1.25, 1.6, boxstyle="round,pad=0.1",
                          facecolor="#1b4332", edgecolor="#52b788", linewidth=2)
    ax.add_patch(fbox)
    ax.text(9.225, 2.7, "✅", ha="center", va="center", fontsize=16)
    ax.text(9.225, 2.2, "Class A", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#d8f3dc")
    ax.annotate("", xy=(8.6, 2.5), xytext=(8.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#1b4332", lw=2))
    ax.text(5, 0.5, "Multiple trees vote → majority wins → final prediction",
            ha="center", va="center", fontsize=9, color="#6c9e82", style="italic")

    st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════
# SECTION 2 — CLASSIFICATION
# ════════════════════════════════════════════════

elif section == "🤖 Random Forest Classification":

    # ── Load dataset ────────────────────────────
    is_titanic = uploaded_file is None

    if is_titanic:
        df_raw = load_titanic()
        target_col = "Survived"
        dataset_name = "Titanic"
        st.sidebar.success("Built-in Titanic dataset loaded ✓")
    else:
        try:
            df_raw = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name.replace(".csv", "")
            st.sidebar.success(f"Custom dataset loaded ✓  ({df_raw.shape[0]} rows)")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Auto-detect target column
        auto_target = detect_target_column(df_raw)
        all_cols = df_raw.columns.tolist()

        st.sidebar.markdown("### 🎯 Target Column")
        target_col = st.sidebar.selectbox(
            "Select Target Column",
            options=all_cols,
            index=all_cols.index(auto_target) if auto_target in all_cols else 0,
            help="Auto-detected. Override if needed."
        )
        if auto_target and auto_target == target_col:
            st.sidebar.caption(f"Auto-detected: **{auto_target}**")

        # Validate
        is_valid, reason = validate_classification_dataset(df_raw, target_col)
        if not is_valid:
            show_invalid_dataset_message(reason)
            st.stop()

    # ── Hero subtitle update ─────────────────────
    st.markdown(f"""
    <div style="
        background: rgba(64,145,108,0.12); border: 1px solid #74c69d;
        border-radius: 10px; padding: 10px 18px; margin-bottom: 20px;
        font-size: 0.92rem; color: #2d6a4f; font-weight: 500;
    ">
        📂 Active dataset: <strong>{dataset_name}</strong> &nbsp;·&nbsp;
        🎯 Target column: <strong>{target_col}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset overview ─────────────────────────
    st.markdown('<div class="section-heading">📊 Dataset Overview</div>', unsafe_allow_html=True)

    target_series = df_raw[target_col].dropna()
    class_balance = target_series.value_counts()
    n_classes = class_balance.nunique() if False else target_series.nunique()
    missing_total = df_raw.isnull().sum().sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_raw.shape[0])
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Target Classes", n_classes)
    col4.metric("Missing Cells", missing_total)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Dataset Preview**")
        st.dataframe(df_raw.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Statistical Summary**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Column Guide ─────────────────────────────
    st.markdown('<div class="section-heading">📘 Column Guide</div>', unsafe_allow_html=True)

    cols3 = st.columns(3)
    for i, col in enumerate(df_raw.columns):
        dtype = str(df_raw[col].dtype)
        missing = int(df_raw[col].isnull().sum())
        nuniq = int(df_raw[col].nunique())
        role = "🎯 Target" if col == target_col else "📐 Feature"
        icon = "🎯" if col == target_col else ("🔢" if pd.api.types.is_numeric_dtype(df_raw[col]) else "🔤")
        with cols3[i % 3]:
            st.markdown(f"""
            <div class="col-info-card">
                <div class="col-name">{icon} {col} <span style="font-weight:400;color:#6c9e82;font-size:0.78rem;">({role})</span></div>
                <div class="col-desc">
                    Type: <b>{dtype}</b> · Missing: <b>{missing}</b> · Unique values: <b>{nuniq}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Missing value handling ───────────────────
    st.markdown('<div class="section-heading">🧹 Missing Value Handling</div>', unsafe_allow_html=True)

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Before Handling**")
        st.dataframe(df_raw.isnull().sum().rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Build working copy with missing values filled
    df_filled = df_raw.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        else:
            mode_val = df_filled[col].mode()
            df_filled[col] = df_filled[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    with col_after:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**After Handling**")
        st.dataframe(df_filled.isnull().sum().rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View handling code"):
        st.code("""
# Numerical → median, Categorical → mode
for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
""", language="python")

    # ── Outlier visualisation ────────────────────
    st.markdown('<div class="section-heading">📦 Outlier Visualisation</div>', unsafe_allow_html=True)

    numeric_cols = df_filled.select_dtypes(include=np.number).columns.tolist()
    df_no_outliers = remove_outliers(df_filled, target_col)

    tab_before, tab_after = st.tabs(["Before Removal", "After Removal"])

    with tab_before:
        if numeric_cols:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.boxplot(data=df_filled[numeric_cols], ax=ax,
                        palette=["#74c69d"] * len(numeric_cols))
            plt.xticks(rotation=35, fontsize=8)
            ax.set_title("Outliers Before Handling", fontsize=11, color="#2d6a4f", pad=10)
            apply_plot_theme(fig, [ax])
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns to display.")

    with tab_after:
        numeric_cols_after = df_no_outliers.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_after:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.boxplot(data=df_no_outliers[numeric_cols_after], ax=ax,
                        palette=["#52b788"] * len(numeric_cols_after))
            plt.xticks(rotation=35, fontsize=8)
            ax.set_title("Outliers After Handling", fontsize=11, color="#2d6a4f", pad=10)
            apply_plot_theme(fig, [ax])
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns to display.")

    # ── Preprocessing pipeline ───────────────────
    st.markdown('<div class="section-heading">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)

    try:
        X_train, X_test, y_train, y_test, scaler, X, y, drop_cols, class_names = prepare_model_data(
            df_no_outliers, target_col
        )
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features",          X.shape[1])
    col2.metric("Training Samples",  X_train.shape[0])
    col3.metric("Test Samples",       X_test.shape[0])
    col4.metric("Target Classes",     y.nunique())

    if drop_cols:
        st.markdown(
            f"<small style='color:#6c9e82;'>🗑️ Dropped columns: {', '.join(drop_cols)}</small>",
            unsafe_allow_html=True
        )

    # ── Model training ───────────────────────────
    st.markdown('<div class="section-heading">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    col_acc, col_cm = st.columns([1, 2])

    with col_acc:
        st.metric("✅ Accuracy Score", f"{accuracy*100:.1f}%")
        st.markdown("")
        st.markdown("**Classification Report**")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        rdf = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(rdf, use_container_width=True)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        # Use class names if they fit (up to 10 classes)
        if len(class_names) <= 10:
            tick_labels = [str(c) for c in class_names]
        else:
            tick_labels = [str(c) for c in sorted(y_test.unique())]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette("#40916c", as_cmap=True),
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    ax=ax, linewidths=0.5, linecolor="#d8f3dc")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title("Confusion Matrix", fontsize=11, color="#2d6a4f")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    # ── Feature importance ───────────────────────
    st.markdown('<div class="section-heading">⭐ Feature Importance</div>', unsafe_allow_html=True)

    importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(importance) * 0.38)))
    colors = sns.light_palette("#40916c", n_colors=len(importance))
    ax.barh(importance["Feature"], importance["Importance"],
            color=colors, edgecolor="none", height=0.65)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("Feature Importance Ranking", fontsize=11, color="#2d6a4f", pad=10)
    for i, (val, name) in enumerate(zip(importance["Importance"], importance["Feature"])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=7.5, color="#2d6a4f")
    apply_plot_theme(fig, [ax])
    st.pyplot(fig, use_container_width=True)

    # ── Tree depth visualisation ─────────────────
    st.markdown(
        '<div class="section-heading">🌳 Interactive Tree Depth Visualisation</div>',
        unsafe_allow_html=True
    )

    depth = st.slider("🎚️ Drag to change tree depth", min_value=1, max_value=8, value=3, step=1)

    depth_labels = {
        1: ("Very simple tree.", "⚡ Fast but may underfit"),
        2: ("Simple structure.", "Good for prototypes"),
        3: ("Balanced depth.", "✅ Recommended"),
        4: ("Captures more patterns.", "Slightly complex"),
        5: ("Complex tree.", "Watch overfitting"),
        6: ("Very deep tree.", "⚠️ High variance"),
        7: ("Highly complex.", "⚠️ Overfitting risk"),
        8: ("Extreme depth.", "❌ Usually not ideal"),
    }
    label, note = depth_labels[depth]

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);padding:14px;border-radius:12px;
                border-left:5px solid #52b788;margin-bottom:15px;">
    <b>Depth {depth}:</b> {label}<br>
    <span style="color:#95d5b2;">{note}</span>
    </div>
    """, unsafe_allow_html=True)

    model_depth = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    model_depth.fit(X_train, y_train)

    st.markdown("### 🌲 Actual Decision Tree Representation")

    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("#EFF2F6")
    ax.set_facecolor("#16E9B4")

    # Limit class_names for tree display
    tree_class_names = [str(c) for c in class_names] if len(class_names) <= 20 else None

    plot_tree(
        model_depth.estimators_[0],
        feature_names=X.columns,
        class_names=tree_class_names,
        filled=True, rounded=True, fontsize=8,
        max_depth=depth, impurity=True, proportion=True, precision=2, ax=ax
    )
    ax.set_title(f"Random Forest Tree (Depth = {depth})", fontsize=18,
                 color="#16E9B4", pad=20, weight="bold")
    st.pyplot(fig, use_container_width=True)

    # ── Prediction section ───────────────────────
    if is_titanic:
        pred_section_title = "🚢 Passenger Survival Predictor"
    else:
        pred_section_title = f"🔮 {dataset_name} Predictor"

    st.markdown(f'<div class="section-heading">{pred_section_title}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-bottom:20px;">
        <p style="margin:0; font-size:0.93rem; line-height:1.7;">
        Fill in the feature values below and click <strong>Predict</strong>.
        The model uses <strong>tree depth {depth}</strong> from the slider above.
        The target column is <strong>{target_col}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Build dynamic form
    input_scaled = build_prediction_form(df_raw, target_col, X.columns, scaler)

    if input_scaled is not None and st.button("🔮 Predict"):
        prediction  = model_depth.predict(input_scaled)[0]
        proba       = model_depth.predict_proba(input_scaled)[0]
        pred_class  = class_names[prediction] if prediction < len(class_names) else prediction
        confidence  = float(proba.max()) * 100

        # Determine if binary for survival-style display
        is_binary = len(class_names) == 2

        if is_binary:
            pos_class_idx = 1
            pos_prob = float(proba[pos_class_idx]) * 100 if pos_class_idx < len(proba) else confidence

            if prediction == pos_class_idx:
                st.markdown(f"""
                <div class="result-survived">
                    <h2>✅ Predicted: {pred_class}</h2>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-survived">
                    <h2>❌ Predicted: {pred_class}</h2>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:16px;">
                <div style="font-size:0.85rem;font-weight:600;margin-bottom:4px;color:#6c9e82;">
                    Confidence: {class_names[pos_class_idx] if pos_class_idx < len(class_names) else "Positive"}
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{pos_prob}%;"></div>
                </div>
                <div style="font-size:0.8rem;color:#6c9e82;text-align:right;">{pos_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Multi-class
            st.markdown(f"""
            <div class="result-survived">
                <h2>✅ Predicted: {pred_class}</h2>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Show all class probabilities
            st.markdown("**Class Probabilities**")
            prob_df = pd.DataFrame({
                "Class": [str(class_names[i]) if i < len(class_names) else str(i) for i in range(len(proba))],
                "Probability": [f"{p*100:.1f}%" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True)

        # Tree voting summary
        all_votes = [tree.predict(input_scaled)[0] for tree in model_depth.estimators_]
        st.markdown("")
        if is_binary:
            pos_votes = int(np.sum([v == 1 for v in all_votes]))
            neg_votes = len(all_votes) - pos_votes
            cv1, cv2 = st.columns(2)
            cv1.metric(f"🌲 Trees voting '{class_names[1] if len(class_names) > 1 else 1}'", pos_votes)
            cv2.metric(f"🌲 Trees voting '{class_names[0] if len(class_names) > 0 else 0}'", neg_votes)
        else:
            from collections import Counter
            vote_counts = Counter(all_votes)
            vote_cols = st.columns(min(len(vote_counts), 4))
            for i, (cls, cnt) in enumerate(sorted(vote_counts.items())):
                cls_name = str(class_names[cls]) if cls < len(class_names) else str(cls)
                with vote_cols[i % len(vote_cols)]:
                    st.metric(f"🌲 '{cls_name}'", cnt)
