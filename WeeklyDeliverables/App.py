import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from collections import Counter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark clinical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}
.pulse-dot {
    display: inline-block;
    width: 9px; height: 9px;
    background: #ef4444;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1f2937;
}

/* Metric cards */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1; min-width: 130px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Info boxes */
.info-box {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
}
.info-box strong { color: #c7d2fe; }

/* Hypothesis badge */
.badge-yes {
    background: #052e16; color: #4ade80;
    border: 1px solid #166534;
    padding: 2px 10px; border-radius: 99px; font-size: 0.75rem;
}
.badge-no {
    background: #1c1917; color: #94a3b8;
    border: 1px solid #292524;
    padding: 2px 10px; border-radius: 99px; font-size: 0.75rem;
}

/* Tabs */
[data-baseweb="tab-list"] { background: #111827 !important; border-radius: 8px; gap: 4px; }
[data-baseweb="tab"] { background: transparent !important; color: #64748b !important; border-radius: 6px !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1e1b4b !important; color: #a5b4fc !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1f2937; border-radius: 8px; }

/* Plot background helper */
.plot-wrap {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Selectbox / sliders */
[data-baseweb="select"] { background: #111827 !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "text.color": "#e2e8f0",
    "grid.color": "#1f2937",
    "grid.linewidth": 0.6,
    "legend.facecolor": "#0f172a",
    "legend.edgecolor": "#1f2937",
    "font.family": "monospace",
})

ACCENT = "#6366f1"
ACCENT2 = "#f43f5e"
ACCENT3 = "#34d399"
PALETTE = [ACCENT, ACCENT2, ACCENT3, "#f59e0b", "#38bdf8", "#e879f9"]

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("dataset_processed.csv")
    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    df_model = df.drop(columns=["id", "dataset"])
    df_model["fbs"] = df_model["fbs"].astype(int)
    df_model["exang"] = df_model["exang"].astype(int)
    df_enc = pd.get_dummies(df_model, columns=["sex", "cp", "restecg"], drop_first=True)
    df_enc = df_enc.apply(lambda c: c.astype(int) if c.dtype == bool else c)

    X = df_enc.drop(columns=["num"])
    y = df_enc["num"]
    return df, df_model, X, y

df, df_model, X, y = load_and_prepare()

@st.cache_data
def run_stats(df, X, y):
    group0 = df[df["num"] == 0]
    group1 = df[df["num"] == 1]
    alpha = 0.05
    results = []
    for feat in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
        stat, p = mannwhitneyu(group0[feat].dropna(), group1[feat].dropna(), alternative="two-sided")
        results.append({"Feature": feat, "Test": "Mann-Whitney U", "Statistic": round(stat, 1), "p-value": p, "Significant": p < alpha})
    for feat in ["sex", "cp", "fbs", "restecg", "exang"]:
        ct = pd.crosstab(df[feat], df["num"])
        c2, p, _, _ = chi2_contingency(ct)
        results.append({"Feature": feat, "Test": "Chi-Square", "Statistic": round(c2, 2), "p-value": p, "Significant": p < alpha})
    return pd.DataFrame(results).sort_values("p-value")

@st.cache_data
def run_feature_selection(X_arr, y_arr, cols):
    X_arr = np.array(X_arr); y_arr = np.array(y_arr)
    sel = SelectKBest(f_classif, k=10).fit(X_arr, y_arr)
    kbest = set(np.array(cols)[sel.get_support()])
    lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42, max_iter=500)
    lr_l1.fit(X_arr, y_arr)
    l1_feats = set(np.array(cols)[lr_l1.coef_[0] != 0])
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_arr, y_arr)
    imp = pd.Series(rf.feature_importances_, index=cols)
    rf_feats = set(imp.nlargest(10).index)
    count = Counter(list(kbest) + list(l1_feats) + list(rf_feats))
    consensus = [f for f, c in count.items() if c >= 2]
    return sel, lr_l1, rf, imp, consensus, count

@st.cache_data
def run_model(X_arr, y_arr, cols, consensus):
    X_c = X_arr[:, [list(cols).index(f) for f in consensus]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_c)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(Xtr, ytr)
    ypred = lr.predict(Xte)
    yprob = lr.predict_proba(Xte)[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(lr, Xs, y_arr, cv=cv, scoring="roc_auc")
    # Statsmodels for odds ratios
    Xsm = sm.add_constant(Xs)
    try:
        sm_res = sm.Logit(y_arr, Xsm).fit(maxiter=200, disp=False)
    except Exception:
        sm_res = None
    return lr, yprob, ypred, yte, cv_auc, sm_res, Xs, scaler

X_arr = X.values.astype(float)
y_arr = y.values.astype(int)
cols = list(X.columns)

stats_df = run_stats(df, X, y)
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_arr)

sel, lr_l1, rf_model, rf_imp, consensus, feat_count = run_feature_selection(X_scaled_full, y_arr, cols)
lr_model, y_prob, y_pred, y_test, cv_auc, sm_res, X_final_scaled, consensus_scaler = run_model(X_arr, y_arr, cols, consensus)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 Navigation")
    page = st.radio("Navigate", [
        "📊 Overview & EDA",
        "🧪 Hypothesis Testing",
        "🔬 Feature Selection",
        "🤖 Model & Results",
        "🔍 Predict",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- **Rows:** {len(df)}")
    st.markdown(f"- **Features:** {len(df.columns) - 1}")
    st.markdown(f"- **Target:** `num` (binary)")
    disease_pct = df['num'].mean() * 100
    st.markdown(f"- **Disease rate:** {disease_pct:.1f}%")

    st.markdown("---")
    st.markdown('<div style="color:#374151;font-size:0.75rem;">Heart Disease Prediction · Statistical Analysis Dashboard</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1><span class="pulse-dot"></span>Heart Disease · Statistical Analysis</h1>
  <p>EDA · Hypothesis Testing · Feature Selection · Logistic Regression · Model Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════
if page == "📊 Overview & EDA":

    st.markdown('<div class="section-label">Section 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview & Exploratory Analysis</div>', unsafe_allow_html=True)

    # KPI cards
    n0 = (df["num"] == 0).sum()
    n1 = (df["num"] == 1).sum()
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Total Patients</div></div>
        <div class="metric-card"><div class="val">{n0}</div><div class="lbl">No Disease</div></div>
        <div class="metric-card"><div class="val">{n1}</div><div class="lbl">Disease</div></div>
        <div class="metric-card"><div class="val">{n1/len(df)*100:.1f}%</div><div class="lbl">Prevalence</div></div>
        <div class="metric-card"><div class="val">{df['age'].mean():.1f}</div><div class="lbl">Avg Age</div></div>
        <div class="metric-card"><div class="val">{(df['sex']=='Male').mean()*100:.0f}%</div><div class="lbl">Male</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📈 Distributions", "📦 By Target", "🔗 Correlations"])

    with tab1:
        st.dataframe(df.head(50), width="stretch", height=400)
        st.caption(f"Showing 50 of {len(df)} rows")

    with tab2:
        num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
        sel_col = st.selectbox("Select numerical feature", num_cols)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        # Distribution
        axes[0].hist(df[sel_col].dropna(), bins=30, color=ACCENT, alpha=0.85, edgecolor="#0b0f1a")
        from scipy.stats import gaussian_kde
        xs = np.linspace(df[sel_col].min(), df[sel_col].max(), 200)
        kde = gaussian_kde(df[sel_col].dropna())
        ax2 = axes[0].twinx()
        ax2.plot(xs, kde(xs), color=ACCENT2, lw=2)
        ax2.set_yticks([])
        ax2.tick_params(colors="#1f2937")
        ax2.set_facecolor("#111827")
        axes[0].set_title(f"Distribution: {sel_col}", color="#e2e8f0", fontsize=11)
        axes[0].set_xlabel(sel_col)
        axes[0].grid(True, alpha=0.3)

        # Boxplot
        bp = axes[1].boxplot(df[sel_col].dropna(), patch_artist=True, widths=0.4,
                              boxprops=dict(facecolor=ACCENT, color=ACCENT2, alpha=0.7),
                              medianprops=dict(color="#fbbf24", lw=2.5),
                              whiskerprops=dict(color="#94a3b8"),
                              capprops=dict(color="#94a3b8"),
                              flierprops=dict(marker="o", color=ACCENT2, alpha=0.4, markersize=3))
        axes[1].set_title(f"Boxplot: {sel_col}", color="#e2e8f0", fontsize=11)
        axes[1].set_xticks([])
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stats summary
        desc = df[sel_col].describe()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{desc['mean']:.2f}")
        c2.metric("Std", f"{desc['std']:.2f}")
        c3.metric("Min", f"{desc['min']:.1f}")
        c4.metric("Max", f"{desc['max']:.1f}")

    with tab3:
        num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
        sel_feat = st.selectbox("Feature vs Target", num_cols, key="vs")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Violin by target
        parts = axes[0].violinplot([df[df["num"]==0][sel_feat].dropna(), df[df["num"]==1][sel_feat].dropna()],
                                    showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([ACCENT3, ACCENT2][i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("#fbbf24")
        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(["No Disease", "Disease"])
        axes[0].set_title(f"{sel_feat} by Heart Disease", color="#e2e8f0", fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Mean of sel_feat grouped by sex and disease status
        group_data = df.groupby(["sex", "num"])[sel_feat].mean().unstack(fill_value=0)
        group_data.plot(kind="bar", ax=axes[1], color=[ACCENT3, ACCENT2], alpha=0.85, width=0.6, edgecolor="#0b0f1a")
        axes[1].set_title(f"Mean {sel_feat} by Sex & Disease", color="#e2e8f0", fontsize=11)
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(["No Disease", "Disease"], framealpha=0.3)
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "num"]
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                    ax=ax, linewidths=0.5, linecolor="#0b0f1a",
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 10})
        ax.set_title("Correlation Matrix (Numerical Features)", color="#e2e8f0", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
        <strong>Key correlations with target (num):</strong> <code>thalch</code> (max heart rate) shows a negative correlation with disease — lower heart rate capacity is associated with disease. <code>oldpeak</code> and <code>age</code> show positive correlations.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 2 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════
elif page == "🧪 Hypothesis Testing":

    st.markdown('<div class="section-label">Section 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hypothesis Development & Statistical Testing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Approach:</strong> We test each feature for association with the binary target <code>num</code> (heart disease).
    <strong>Mann-Whitney U</strong> is used for continuous variables (no normality assumption required for medical data).
    <strong>Chi-Square</strong> is used for categorical variables. Significance level α = 0.05.
    </div>
    """, unsafe_allow_html=True)

    # Hypothesis table
    hyp_data = [
        ("age",      "Mann-Whitney U", "Age differs between groups",      "Older patients have higher disease risk"),
        ("thalch",   "Mann-Whitney U", "Max HR differs between groups",   "Disease patients have lower max heart rate"),
        ("oldpeak",  "Mann-Whitney U", "Oldpeak differs between groups",  "Higher oldpeak → higher disease probability"),
        ("trestbps", "Mann-Whitney U", "Resting BP differs",              "Higher BP associated with disease"),
        ("chol",     "Mann-Whitney U", "Cholesterol differs",             "Cholesterol distribution differs by disease"),
        ("sex",      "Chi-Square",     "Sex independent of disease",      "Sex significantly associated with disease"),
        ("cp",       "Chi-Square",     "Chest pain type independent",     "Chest pain type strongly predicts disease"),
        ("exang",    "Chi-Square",     "Exercise angina independent",     "Exang significantly predicts disease"),
        ("fbs",      "Chi-Square",     "Fasting BS independent",          "Fasting blood sugar independent of disease"),
        ("restecg",  "Chi-Square",     "ECG result independent",          "ECG abnormalities associated with disease"),
    ]
    hyp_df = pd.DataFrame(hyp_data, columns=["Feature", "Test", "H₀ (Null)", "H₁ (Alternative)"])
    st.dataframe(hyp_df, width="stretch", hide_index=True)

    st.markdown("### Test Results")

    # Results table
    display_df = stats_df.copy()
    display_df["p-value"] = display_df["p-value"].apply(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
    display_df["Decision"] = display_df["Significant"].apply(lambda s: "Reject H₀ ✓" if s else "Fail to Reject ✗")
    display_df = display_df.drop(columns=["Significant"])
    st.dataframe(display_df, width="stretch", hide_index=True)

    # p-value bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    pvals = stats_df["p-value"].values
    feats = stats_df["Feature"].values
    colors = [ACCENT if s else "#374151" for s in stats_df["Significant"].values]

    bars = ax.barh(feats, -np.log10(np.clip(pvals, 1e-20, 1)), color=colors, height=0.6, edgecolor="#0b0f1a")
    ax.axvline(-np.log10(0.05), color=ACCENT2, linestyle="--", lw=1.5, label="α = 0.05")
    ax.set_xlabel("−log₁₀(p-value) →  more significant", color="#94a3b8")
    ax.set_title("Feature Significance (Hypothesis Tests)", color="#e2e8f0", fontsize=13)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="info-box">
    <strong>Result Summary:</strong> All features except <code>fbs</code> (fasting blood sugar) rejected H₀ at α = 0.05.
    <code>cp</code> (chest pain type), <code>thalch</code> (max heart rate), and <code>exang</code> (exercise angina) show the strongest associations with heart disease.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 3 — FEATURE SELECTION
# ══════════════════════════════════════════════
elif page == "🔬 Feature Selection":

    st.markdown('<div class="section-label">Section 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Selection — Tri-Method Consensus</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Three complementary methods are applied:</strong><br>
    <strong>1. SelectKBest</strong> (ANOVA F-test) — ranks features by statistical separability<br>
    <strong>2. L1 Logistic Regression (Lasso)</strong> — shrinks irrelevant feature coefficients to zero<br>
    <strong>3. Random Forest Importance</strong> — tree-based non-linear importance<br><br>
    <strong>Consensus rule:</strong> A feature is selected if it appears in ≥ 2 of 3 methods.
    </div>
    """, unsafe_allow_html=True)

    # VIF check
    st.markdown("#### Multicollinearity Check (VIF)")
    vif_vals = [variance_inflation_factor(X_scaled_full, i) for i in range(X_scaled_full.shape[1])]
    vif_df = pd.DataFrame({"Feature": cols, "VIF": np.round(vif_vals, 2)}).sort_values("VIF", ascending=False)
    vif_df["Status"] = vif_df["VIF"].apply(lambda v: "⚠️ High" if v > 10 else ("⚡ Moderate" if v > 5 else "✅ OK"))
    st.dataframe(vif_df.head(15), width="stretch", hide_index=True)

    st.markdown("#### Feature Importance Comparison")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # SelectKBest
    kbest_scores = pd.Series(sel.scores_, index=cols).sort_values(ascending=False).head(12)
    colors1 = [ACCENT if f in consensus else "#374151" for f in kbest_scores.index]
    axes[0].barh(kbest_scores.index[::-1], kbest_scores.values[::-1], color=colors1[::-1], height=0.6, edgecolor="#0b0f1a")
    axes[0].set_title("SelectKBest\n(F-Score)", color="#e2e8f0", fontsize=11)
    axes[0].grid(True, axis="x", alpha=0.3)

    # L1 Logistic
    l1_coefs = pd.Series(np.abs(lr_l1.coef_[0]), index=cols).sort_values(ascending=False).head(12)
    colors2 = [ACCENT2 if f in consensus else "#374151" for f in l1_coefs.index]
    axes[1].barh(l1_coefs.index[::-1], l1_coefs.values[::-1], color=colors2[::-1], height=0.6, edgecolor="#0b0f1a")
    axes[1].set_title("L1 Lasso\n(|Coefficient|)", color="#e2e8f0", fontsize=11)
    axes[1].grid(True, axis="x", alpha=0.3)

    # Random Forest
    rf_top = rf_imp.sort_values(ascending=False).head(12)
    colors3 = [ACCENT3 if f in consensus else "#374151" for f in rf_top.index]
    axes[2].barh(rf_top.index[::-1], rf_top.values[::-1], color=colors3[::-1], height=0.6, edgecolor="#0b0f1a")
    axes[2].set_title("Random Forest\n(Importance)", color="#e2e8f0", fontsize=11)
    axes[2].grid(True, axis="x", alpha=0.3)

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Highlighted = Consensus Features", color="#64748b", fontsize=10, y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Consensus feature heatmap
    st.markdown("#### Consensus Features")
    count_df = pd.DataFrame([{"Feature": f, "SelectKBest": f in set(cols[i] for i in range(len(cols)) if sel.get_support()[i]),
                               "L1-LR": f in set(np.array(cols)[lr_l1.coef_[0] != 0]),
                               "RandomForest": f in set(rf_imp.nlargest(10).index),
                               "Votes": feat_count.get(f, 0)} for f in set(list(feat_count.keys()))])
    count_df = count_df[count_df["Votes"] >= 1].sort_values("Votes", ascending=False)
    count_df["Selected"] = count_df["Votes"] >= 2
    st.dataframe(count_df, width="stretch", hide_index=True)

    st.success(f"✅ **Consensus features ({len(consensus)}):** {', '.join(sorted(consensus))}")

# ══════════════════════════════════════════════
# PAGE 4 — MODEL & RESULTS
# ══════════════════════════════════════════════
elif page == "🤖 Model & Results":

    st.markdown('<div class="section-label">Section 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Logistic Regression — Statistical Inference & Evaluation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Why Logistic Regression?</strong> The target is binary (disease / no disease). Logistic regression directly models the <em>probability</em> of disease, produces interpretable <strong>odds ratios</strong> per feature, and is the clinical gold standard for this type of problem. We use <code>statsmodels</code> for inference (p-values, CIs) and <code>sklearn</code> for predictive evaluation.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📐 Odds Ratios", "📊 Performance Metrics", "📉 ROC & Confusion Matrix"])

    with tab1:
        if sm_res is not None:
            feat_names = consensus
            coef = sm_res.params[1:]
            pvals = sm_res.pvalues[1:]
            ci_raw = sm_res.conf_int()
            # conf_int() may return ndarray or DataFrame depending on statsmodels version
            if hasattr(ci_raw, "iloc"):
                ci = ci_raw.iloc[1:]
                ci_vals = ci.values
            else:
                ci = ci_raw[1:]
                ci_vals = ci
            or_df = pd.DataFrame({
                "Feature": feat_names[:len(coef)],
                "Coeff": np.round(np.asarray(coef).flatten(), 4),
                "Odds Ratio": np.round(np.exp(np.asarray(coef).flatten()), 4),
                "CI Lower": np.round(np.exp(ci_vals[:, 0]), 4),
                "CI Upper": np.round(np.exp(ci_vals[:, 1]), 4),
                "p-value": np.round(np.asarray(pvals).flatten(), 5),
            })
            or_df["Sig"] = or_df["p-value"].apply(lambda p: "✓" if p < 0.05 else "")
            or_df = or_df.sort_values("Odds Ratio", ascending=False)
            st.dataframe(or_df, width="stretch", hide_index=True)

            # Forest plot
            fig, ax = plt.subplots(figsize=(9, max(4, len(or_df) * 0.5)))
            y_pos = np.arange(len(or_df))
            err_low = or_df["Odds Ratio"].values - or_df["CI Lower"].values
            err_high = or_df["CI Upper"].values - or_df["Odds Ratio"].values
            colors = [ACCENT2 if p < 0.05 else "#374151" for p in or_df["p-value"]]

            ax.barh(y_pos, or_df["Odds Ratio"].values, xerr=[err_low, err_high],
                    color=colors, capsize=4, height=0.55, edgecolor="#0b0f1a", error_kw={"ecolor": "#94a3b8"})
            ax.axvline(1.0, color="#fbbf24", linestyle="--", lw=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(or_df["Feature"].values, fontsize=9)
            ax.set_xlabel("Odds Ratio (with 95% CI)")
            ax.set_title("Forest Plot — Logistic Regression Odds Ratios", color="#e2e8f0", fontsize=12)
            sig_patch = mpatches.Patch(color=ACCENT2, label="p < 0.05 (significant)")
            ns_patch = mpatches.Patch(color="#374151", label="p ≥ 0.05")
            ax.legend(handles=[sig_patch, ns_patch])
            ax.grid(True, axis="x", alpha=0.3)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Statsmodels model did not converge. Try adjusting features.")

    with tab2:
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"], output_dict=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="val">{auc:.3f}</div><div class="lbl">ROC-AUC</div></div>
            <div class="metric-card"><div class="val">{cv_auc.mean():.3f}</div><div class="lbl">CV AUC (5-fold)</div></div>
            <div class="metric-card"><div class="val">±{cv_auc.std():.3f}</div><div class="lbl">CV Std Dev</div></div>
            <div class="metric-card"><div class="val">{report['accuracy']:.3f}</div><div class="lbl">Accuracy</div></div>
            <div class="metric-card"><div class="val">{report['Disease']['f1-score']:.3f}</div><div class="lbl">F1 (Disease)</div></div>
        </div>
        """, unsafe_allow_html=True)

        rep_df = pd.DataFrame(report).T.iloc[:3].round(3)
        st.dataframe(rep_df, width="stretch")

        # CV scores bar
        fig, ax = plt.subplots(figsize=(7, 3))
        fold_colors = [ACCENT if v >= cv_auc.mean() else ACCENT2 for v in cv_auc]
        ax.bar([f"Fold {i+1}" for i in range(5)], cv_auc, color=fold_colors, alpha=0.85, edgecolor="#0b0f1a")
        ax.axhline(cv_auc.mean(), color="#fbbf24", linestyle="--", lw=1.5, label=f"Mean = {cv_auc.mean():.3f}")
        ax.set_ylim(0.7, 1.0)
        ax.set_title("5-Fold Cross Validation AUC", color="#e2e8f0", fontsize=11)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        im = axes[0].imshow(cm, cmap="Blues", vmin=0)
        axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
        axes[0].set_xticklabels(["No Disease", "Disease"]); axes[0].set_yticklabels(["No Disease", "Disease"])
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion Matrix", color="#e2e8f0", fontsize=12)
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i,j]), ha="center", va="center",
                             color="white" if cm[i,j] > cm.max()/2 else "#0b0f1a", fontsize=18, fontweight="bold")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[1].plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"Logistic Reg (AUC = {auc:.3f})")
        axes[1].fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
        axes[1].plot([0,1],[0,1], color="#374151", linestyle="--", lw=1.5, label="Random (AUC = 0.500)")
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve", color="#e2e8f0", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].spines[["top","right"]].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════
elif page == "🔍 Predict":

    st.markdown('<div class="section-label">Section 05</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Single Patient Risk Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter patient data below. The model will output the <strong>probability of heart disease</strong> and a risk classification based on the trained Logistic Regression model.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)

    with col2:
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])
        restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
        thalch = st.slider("Max Heart Rate Achieved", 60, 210, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [False, True])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)

    if st.button("🫀 Predict Risk", width="stretch"):
        # Build input row
        input_row = {
            "age": age, "trestbps": trestbps, "chol": chol, "thalch": thalch,
            "oldpeak": oldpeak, "fbs": int(fbs), "exang": int(exang),
            "sex": sex, "cp": cp, "restecg": restecg
        }
        inp_df = pd.DataFrame([input_row])
        inp_df = pd.get_dummies(inp_df, columns=["sex", "cp", "restecg"], drop_first=True)

        # Align columns
        for col_name in X.columns:
            if col_name not in inp_df.columns:
                inp_df[col_name] = 0
        inp_df = inp_df[X.columns]
        inp_arr = inp_df.values.astype(float)

        # Select consensus features and scale using the SAME scaler fitted during training
        inp_consensus = inp_arr[:, [cols.index(f) for f in consensus]]
        inp_scaled = consensus_scaler.transform(inp_consensus)

        prob = lr_model.predict_proba(inp_scaled)[0][1]
        risk = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        risk_color = "#ef4444" if prob >= 0.5 else "#34d399"

        st.markdown(f"""
        <div style="background:#111827;border:2px solid {risk_color};border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:2.5rem;color:{risk_color};font-weight:700;">{prob*100:.1f}%</div>
            <div style="font-size:1.1rem;color:{risk_color};margin:0.4rem 0;">{risk}</div>
            <div style="color:#64748b;font-size:0.85rem;">Probability of Heart Disease</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([0], [1], color="#1f2937", height=0.5)
        color = "#ef4444" if prob >= 0.5 else "#34d399"
        ax.barh([0], [prob], color=color, height=0.5, alpha=0.85)
        ax.axvline(0.5, color="#fbbf24", lw=2, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50% (threshold)", "75%", "100%"])
        ax.set_title("Risk Probability Gauge", color="#e2e8f0")
        ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()