# ================================
# RANDOM FOREST CLASSIFICATION APP
# TITANIC DATASET - STREAMLIT APP
# Enhanced UI Version
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

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
    page_title="Random Forest · Titanic",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS — Mid-level professional, dark-mode compatible
# ================================

st.markdown("""
<style>

/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --forest-green:   #2d6a4f;
    --leaf-green:     #40916c;
    --mint:           #74c69d;
    --bark:           #6b4226;
    --sand:           #f4ede4;
    --accent:         #52b788;
    --warning:        #e07a5f;
    --radius-card:    14px;
    --radius-sm:      8px;
    --shadow-card:    0 2px 16px rgba(0,0,0,0.08);
    --shadow-hover:   0 6px 24px rgba(0,0,0,0.13);
    --transition:     0.22s ease;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ── Hero banner ── */
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
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 2.8rem;
    letter-spacing: 6px;
    opacity: 0.25;
}
.hero h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #b7e4c7;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 28px 0 14px 0;
    padding-left: 12px;
    border-left: 4px solid #40916c;
    color: inherit;
    letter-spacing: -0.3px;
}

/* ── Cards ── */
.card {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-card);
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-card);
    transition: box-shadow var(--transition);
}
.card:hover { box-shadow: var(--shadow-hover); }

/* ── Stat chip ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--chip-bg, #f0faf5);
    border: 1px solid #b7e4c7;
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #1b4332;
    margin: 4px;
}

/* ── Concept badge ── */
.concept-badge {
    display: inline-flex;
    align-items: flex-start;
    gap: 10px;
    background: var(--badge-bg, #f6fff9);
    border: 1px solid #b7e4c7;
    border-radius: var(--radius-card);
    padding: 14px 18px;
    margin-bottom: 12px;
    width: 100%;
    box-sizing: border-box;
}
.concept-badge .icon { font-size: 1.4rem; flex-shrink: 0; }
.concept-badge .text { }
.concept-badge .text b { display: block; font-size: 0.92rem; font-weight: 700; margin-bottom: 3px; color: #1b4332; }
.concept-badge .text span { font-size: 0.85rem; color: var(--body-muted, #555); line-height: 1.5; }

/* ── Advantage grid ── */
.adv-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
}
.adv-item {
    display: flex;
    align-items: center;
    gap: 9px;
    background: var(--chip-bg, #f0faf5);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--body-color, #222);
    border: 1px solid #d8f3dc;
}

/* ── Step pill ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 14px;
}
.step-num {
    background: #40916c;
    color: #fff;
    border-radius: 50%;
    min-width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-body b { font-weight: 700; font-size: 0.93rem; display: block; margin-bottom: 3px; color: var(--body-color, #222); }
.step-body span { font-size: 0.84rem; color: var(--body-muted, #555); line-height: 1.5; }

/* ── Analogy box ── */
.analogy-box {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-radius: var(--radius-card);
    padding: 18px 22px;
    margin: 16px 0;
    border-left: 4px solid #40916c;
}
.analogy-box p { margin: 0; color: #1b4332; font-size: 0.92rem; line-height: 1.6; }
.analogy-box b { font-weight: 700; }

/* ── Prediction result ── */
.result-survived {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border: 2px solid #52b788;
    border-radius: var(--radius-card);
    padding: 24px 28px;
    text-align: center;
}
.result-survived h2 { color: #1b4332; margin: 0 0 6px 0; font-size: 1.5rem; }
.result-survived p  { color: #2d6a4f; margin: 0; font-size: 1rem; }

.result-not-survived {
    background: linear-gradient(135deg, #fde8e4, #f9c8bf);
    border: 2px solid #e07a5f;
    border-radius: var(--radius-card);
    padding: 24px 28px;
    text-align: center;
}
.result-not-survived h2 { color: #7a2c1a; margin: 0 0 6px 0; font-size: 1.5rem; }
.result-not-survived p  { color: #9e3b23; margin: 0; font-size: 1rem; }

/* ── Probability bar ── */
.prob-bar-wrap {
    background: rgba(0,0,0,0.08);
    border-radius: 50px;
    height: 10px;
    margin: 12px 0 4px 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #52b788, #2d6a4f);
    transition: width 0.5s ease;
}

/* ── Depth info box ── */
.depth-info {
    background: var(--chip-bg, #f0faf5);
    border: 1px solid #b7e4c7;
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 0.84rem;
    color: var(--body-muted, #444);
    margin-top: 10px;
    line-height: 1.55;
}
.depth-info b { color: #1b4332; }

/* ── Column info card ── */
.col-info-card {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-sm);
    padding: 13px 15px;
    margin-bottom: 10px;
}
.col-info-card .col-name {
    font-weight: 700;
    font-size: 0.88rem;
    color: #2d6a4f;
    margin-bottom: 3px;
}
.col-info-card .col-desc {
    font-size: 0.82rem;
    color: var(--body-muted, #555);
    line-height: 1.45;
}

/* ── Dark mode overrides ── */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1e2a23;
        --card-border: rgba(255,255,255,0.08);
        --chip-bg: #1b3028;
        --badge-bg: #1b3028;
        --body-color: #e8f5ed;
        --body-muted: #9dbfab;
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
}

/* ── Streamlit widget label ── */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-weight: 600;
    font-size: 0.88rem;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #40916c);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 10px 28px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: opacity var(--transition), transform var(--transition);
    box-shadow: 0 3px 10px rgba(45,106,79,0.3);
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(45,106,79,0.35);
}

/* ── Metric card ── */
[data-testid="stMetric"] {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-card);
    padding: 16px 20px;
    box-shadow: var(--shadow-card);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg, 
            );
    border-right: 1px solid var(--card-border, rgba(0,0,0,0.06));
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    font-weight: 500;
    padding: 5px 0;
}

/* ── Table ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-sm);
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ================================
# HERO HEADER
# ================================

st.markdown("""
<div class="hero">
    <h1>🌲 Random Forest Classification</h1>
    <p>Titanic Survival Prediction · Beginner-Friendly Machine Learning App</p>
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
    st.markdown(
        "<small style='color:#6c9e82;'>Built with Streamlit · Scikit-learn</small>",
        unsafe_allow_html=True
    )

# ────────────────────────────────────────────────
# HELPER: matplotlib theme that works in dark mode
# ────────────────────────────────────────────────

def apply_plot_theme(fig, ax_list=None):
    """Minimal neutral theme — readable in both light & dark mode."""
    fig.patch.set_facecolor("none")
    for ax in (ax_list or fig.axes):
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_color("#80f0a7")
        ax.tick_params(colors="#6c9e82", labelsize=9)
        ax.xaxis.label.set_color("#6c9e82")
        ax.yaxis.label.set_color("#6c9e82")
        ax.title.set_color("#2d6a4f")


# ════════════════════════════════════════════════
# SECTION 1 — WHAT IS RANDOM FOREST?
# ════════════════════════════════════════════════

if section == "🌳 What is Random Forest?":

    # ── Simple Explanation ──────────────────────
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

    # ── Real-life analogy ───────────────────────
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

    # ── Core concepts ───────────────────────────
    st.markdown('<div class="section-heading">🧩 Core Concepts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🌲</div>
            <div class="text">
                <b>Decision Tree</b>
                <span>A single model that splits data into branches based on yes/no questions, like a flowchart.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🗳️</div>
            <div class="text">
                <b>Majority Voting</b>
                <span>Each tree votes. The class with the most votes becomes the final prediction.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🎲</div>
            <div class="text">
                <b>Random Sampling</b>
                <span>Each tree trains on a random subset of the data, so every tree is slightly different.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🏡</div>
            <div class="text">
                <b>The Forest</b>
                <span>Many diverse trees together. Diversity is the key — it reduces errors from any single tree.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── How it works ────────────────────────────
    st.markdown('<div class="section-heading">⚙️ How It Works — Step by Step</div>', unsafe_allow_html=True)

    steps = [
        ("Collect Data", "Feed the model your dataset. For Titanic, this includes age, gender, ticket class, fare, and family size."),
        ("Build Many Trees", "The algorithm creates hundreds of decision trees. Each tree sees a slightly different random sample of the data."),
        ("Each Tree Predicts", "For a new passenger, every individual tree makes its own survival prediction independently."),
        ("Vote & Decide", "All trees cast their vote. If 70 out of 100 trees say 'Survived' — the final answer is Survived."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{i}</div>
            <div class="step-body">
                <b>{title}</b>
                <span>{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Advantages ──────────────────────────────
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

    # ── Visual diagram ──────────────────────────
    st.markdown('<div class="section-heading">🎨 Visual: How Random Forest Decides</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Input box
    inp = FancyBboxPatch((0.1, 1.7), 1.6, 1.6,
                         boxstyle="round,pad=0.1",
                         facecolor="#d8f3dc", edgecolor="#40916c", linewidth=2)
    ax.add_patch(inp)
    ax.text(0.9, 2.5, "📊\nInput\nData", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1b4332")

    # Trees
    tree_colors = ["#b7e4c7", "#95d5b2", "#74c69d"]
    tree_labels = ["Tree 1", "Tree 2", "Tree 3"]
    tree_votes  = ["Survived ✓", "Not Survived ✗", "Survived ✓"]
    tree_x = [2.9, 3.9, 4.9]
    for i, (tx, tc, tl, tv) in enumerate(zip(tree_x, tree_colors, tree_labels, tree_votes)):
        box = FancyBboxPatch((tx, 1.2), 1.4, 2.6,
                             boxstyle="round,pad=0.1",
                             facecolor=tc, edgecolor="#40916c", linewidth=1.5)
        ax.add_patch(box)
        ax.text(tx + 0.7, 3.45, f"🌲 {tl}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1b4332")
        ax.text(tx + 0.7, 2.55, "?", ha="center", va="center",
                fontsize=28, color="#2d6a4f", alpha=0.35)
        ax.text(tx + 0.7, 1.58, tv, ha="center", va="center",
                fontsize=7.5, color="#1b4332",
                fontweight="bold")
        # Arrow from input
        ax.annotate("", xy=(tx + 0.05, 2.5), xytext=(1.7, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#40916c",
                                   lw=1.5, connectionstyle="arc3,rad=0"))

    # Voting box
    vbox = FancyBboxPatch((6.8, 1.5), 1.6, 2.0,
                          boxstyle="round,pad=0.1",
                          facecolor="#52b788", edgecolor="#2d6a4f", linewidth=2)
    ax.add_patch(vbox)
    ax.text(7.6, 2.8, "🗳️ Vote", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#fff")
    ax.text(7.6, 2.3, "2 vs 1", ha="center", va="center",
            fontsize=8.5, color="#d8f3dc")

    # Arrows from trees to vote
    for tx in tree_x:
        ax.annotate("", xy=(6.8, 2.5), xytext=(tx + 1.45, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#40916c",
                                   lw=1.5, connectionstyle="arc3,rad=0"))

    # Final result
    fbox = FancyBboxPatch((8.6, 1.7), 1.25, 1.6,
                          boxstyle="round,pad=0.1",
                          facecolor="#1b4332", edgecolor="#52b788", linewidth=2)
    ax.add_patch(fbox)
    ax.text(9.225, 2.7, "✅", ha="center", va="center", fontsize=16)
    ax.text(9.225, 2.2, "Survived", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#d8f3dc")

    ax.annotate("", xy=(8.6, 2.5), xytext=(8.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#1b4332", lw=2))

    ax.text(5, 0.5, "Multiple trees vote → majority wins → final prediction",
            ha="center", va="center", fontsize=9, color="#6c9e82",
            style="italic")

    st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════
# SECTION 2 — RANDOM FOREST CLASSIFICATION
# ════════════════════════════════════════════════

elif section == "🤖 Random Forest Classification":

    # ── Dataset upload ───────────────────────────
    st.sidebar.markdown("### 📂 Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded ✓")
    else:
        df = sns.load_dataset("titanic")
        # Normalise column names to match expected schema
        rename_map = {
            "survived": "Survived",
            "pclass":   "Pclass",
            "sex":      "Sex",
            "age":      "Age",
            "sibsp":    "SibSp",
            "parch":    "Parch",
            "fare":     "Fare",
            "embarked": "Embarked",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        st.sidebar.success("Built-in Titanic dataset loaded ✓")

    # ── Dataset overview ─────────────────────────
    st.markdown('<div class="section-heading">📊 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        surv_rate = int(df["Survived"].mean() * 100) if "Survived" in df.columns else "—"
        st.metric("Survival Rate", f"{surv_rate}%")
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Missing Cells", missing)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Dataset Preview**")
        st.dataframe(df.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Column explanation ───────────────────────
    st.markdown('<div class="section-heading">📘 Column Guide</div>', unsafe_allow_html=True)

    columns_info = {
        "Survived":    ("🎯", "Target column. 0 = Did not survive, 1 = Survived."),
        "Pclass":      ("🎫", "Passenger class — 1st, 2nd, or 3rd. Higher class = more luxury."),
        "Name":        ("👤", "Full passenger name. Dropped before modeling."),
        "Sex":         ("👫", "Male or Female. One of the most important predictors."),
        "Age":         ("🎂", "Passenger age in years. Missing values filled with median."),
        "SibSp":       ("👨‍👩‍👦", "Number of siblings / spouses traveling with passenger."),
        "Parch":       ("👪", "Number of parents / children traveling with passenger."),
        "Ticket":      ("🎟️", "Ticket number. Dropped — too many unique values."),
        "Fare":        ("💰", "Amount paid for the ticket. Higher fare → usually 1st class."),
        "Cabin":       ("🛏️", "Cabin number. Dropped due to many missing values."),
        "Embarked":    ("⚓", "Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton."),
    }

    cols3 = st.columns(3)
    for i, (col_name, (icon, desc)) in enumerate(columns_info.items()):
        with cols3[i % 3]:
            st.markdown(f"""
            <div class="col-info-card">
                <div class="col-name">{icon} {col_name}</div>
                <div class="col-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Missing value handling ───────────────────
    st.markdown('<div class="section-heading">🧹 Missing Value Handling</div>', unsafe_allow_html=True)

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Before Handling**")
        st.dataframe(df.isnull().sum().rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Apply fixes
    df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    for drop_col in ["Cabin", "cabin", "deck"]:
        if drop_col in df.columns:
            df.drop(drop_col, axis=1, inplace=True)

    with col_after:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**After Handling**")
        st.dataframe(df.isnull().sum().rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View handling code"):
        st.code("""
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)
""", language="python")

    # ── Outlier visualisation ────────────────────
    st.markdown('<div class="section-heading">📦 Outlier Visualisation</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_before, tab_after = st.tabs(["Before Removal", "After Removal"])

    with tab_before:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df[numeric_cols], ax=ax,
                    palette=["#74c69d"] * len(numeric_cols))
        plt.xticks(rotation=35, fontsize=8)
        ax.set_title("Outliers Before Handling", fontsize=11, color="#2d6a4f", pad=10)
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    df_outlier = df.copy()
    for col in numeric_cols:
        if col != "Survived":
            Q1, Q3 = df_outlier[col].quantile(0.25), df_outlier[col].quantile(0.75)
            IQR = Q3 - Q1
            df_outlier = df_outlier[
                (df_outlier[col] >= Q1 - 1.5 * IQR) &
                (df_outlier[col] <= Q3 + 1.5 * IQR)
            ]

    with tab_after:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df_outlier[numeric_cols], ax=ax,
                    palette=["#52b788"] * len(numeric_cols))
        plt.xticks(rotation=35, fontsize=8)
        ax.set_title("Outliers After Handling", fontsize=11, color="#2d6a4f", pad=10)
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    # ── Preprocessing ────────────────────────────
    st.markdown('<div class="section-heading">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)

    drop_candidates = ["PassengerId", "Name", "Ticket", "passenger_id", "name", "ticket", "who", "alive", "adult_male", "embark_town", "class", "alone"]
    existing_drops  = [c for c in drop_candidates if c in df_outlier.columns]
    df_model        = df_outlier.drop(existing_drops, axis=1)
    df_model        = pd.get_dummies(df_model, drop_first=True)

    X = df_model.drop("Survived", axis=1)
    y = df_model["Survived"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features",        X.shape[1])
    col2.metric("Training samples", X_train.shape[0])
    col3.metric("Test samples",     X_test.shape[0])
    col4.metric("Target classes",   y.nunique())

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
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette("#40916c", as_cmap=True),
                    xticklabels=["Not Survived", "Survived"],
                    yticklabels=["Not Survived", "Survived"],
                    ax=ax, linewidths=0.5, linecolor="#d8f3dc")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title("Confusion Matrix", fontsize=11, color="#2d6a4f")
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    # ── Feature importance ───────────────────────
    st.markdown('<div class="section-heading">⭐ Feature Importance</div>', unsafe_allow_html=True)

    importance = (
        pd.DataFrame({"Feature": X.columns,
                      "Importance": model.feature_importances_})
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

    # ── Interactive tree depth visualisation ─────
    st.markdown(
        '<div class="section-heading">🌳 Interactive Tree Depth Visualisation</div>',
        unsafe_allow_html=True
    )

    depth = st.slider(
        "🎚️ Drag to change tree depth",
        min_value=1,
        max_value=8,
        value=3,
        step=1
    )

    depth_labels = {
        1: ("Very simple tree.", "⚡ Fast but may underfit"),
        2: ("Simple structure.", "Good for prototypes"),
        3: ("Balanced depth.", "✅ Recommended"),
        4: ("Captures more patterns.", "Slightly complex"),
        5: ("Complex tree.", "Watch overfitting"),
        6: ("Very deep tree.", "⚠️ High variance"),
        7: ("Highly complex.", "⚠️ Overfitting risk"),
        8: ("Extreme depth.", "❌ Usually not ideal")
    }

    label, note = depth_labels[depth]

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.06);
        padding: 14px;
        border-radius: 12px;
        border-left: 5px solid #52b788;
        margin-bottom: 15px;
    ">
    <b>Depth {depth}:</b> {label}<br>
    <span style="color:#95d5b2;">{note}</span>
    </div>
    """, unsafe_allow_html=True)

    # ==========================
    # Train Model
    # ==========================
    model_depth = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model_depth.fit(X_train, y_train)

    # ==========================
    # REAL TREE VISUALIZATION
    # ==========================
    st.markdown("### 🌲 Actual Decision Tree Representation")

    fig, ax = plt.subplots(figsize=(22, 10))

    # DARK MODE
    fig.patch.set_facecolor("#EFF2F6")
    ax.set_facecolor("#16E9B4")

    plot_tree(
        model_depth.estimators_[0],
        feature_names=X.columns,
        class_names=["Not Survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=depth,
        impurity=True,
        proportion=True,
        precision=2,
        ax=ax
    )

    # Title Styling
    ax.set_title(
        f"Random Forest Tree (Depth = {depth})",
        fontsize=18,
        color="#16E9B4",
        pad=20,
        weight="bold"
    )

    st.pyplot(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # PREDICTION SECTION
    # ────────────────────────────────────────────
    st.markdown('<div class="section-heading">🚢 Passenger Survival Predictor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom:20px;">
        <p style="margin:0; font-size:0.93rem; line-height:1.7;">
        Fill in the passenger details below and click <strong>Predict Survival</strong>.
        The model currently uses <strong>tree depth</strong> from the slider above — 
        try adjusting it to see how it affects confidence!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass   = st.selectbox("🎫 Passenger Class", [1, 2, 3],
                                format_func=lambda x: f"Class {x} — {'First' if x==1 else 'Second' if x==2 else 'Third'}")
        sex      = st.selectbox("👫 Sex", ["male", "female"])
        age      = st.slider("🎂 Age", 1, 80, 28)

    with col2:
        fare     = st.number_input("💰 Fare (£)", min_value=0.0, value=32.0, step=1.0)
        sibsp    = st.slider("👨‍👩‍👦 Siblings / Spouses", 0, 8, 0)

    with col3:
        parch    = st.slider("👪 Parents / Children", 0, 6, 0)
        embarked = st.selectbox("⚓ Port of Embarkation",
                                ["S", "C", "Q"],
                                format_func=lambda x: {"S": "S — Southampton", "C": "C — Cherbourg", "Q": "Q — Queenstown"}[x])

    # Build input dataframe aligned to training columns
    input_df = pd.DataFrame({
        "Pclass":      [pclass],
        "Age":         [age],
        "SibSp":       [sibsp],
        "Parch":       [parch],
        "Fare":        [fare],
        "Sex_male":    [1 if sex == "male" else 0],
        "Embarked_Q":  [1 if embarked == "Q" else 0],
        "Embarked_S":  [1 if embarked == "S" else 0],
    })
    input_df    = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    if st.button("🔮 Predict Survival"):
        prediction  = model_depth.predict(input_scaled)[0]
        probability = model_depth.predict_proba(input_scaled)[0][1]
        pct         = probability * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-survived">
                <h2>✅ Likely to Survive</h2>
                <p>Survival probability: <strong>{pct:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-survived">
                <h2>❌ Less Likely to Survive</h2>
                <p>Survival probability: <strong>{pct:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown(f"""
        <div style="margin-top:16px;">
            <div style="font-size:0.85rem; font-weight:600; margin-bottom:4px; color:#6c9e82;">
                Survival Probability
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct}%;"></div>
            </div>
            <div style="font-size:0.8rem; color:#6c9e82; text-align:right;">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Show all tree votes
        all_votes = [tree.predict(input_scaled)[0] for tree in model_depth.estimators_]
        survived_count     = int(np.sum(all_votes))
        not_survived_count = len(all_votes) - survived_count

        st.markdown("")
        col_v1, col_v2 = st.columns(2)
        col_v1.metric("🌲 Trees voting Survived",     survived_count)
        col_v2.metric("🌲 Trees voting Not Survived", not_survived_count)