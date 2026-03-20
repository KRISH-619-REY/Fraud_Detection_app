# ============================================================
# FraudShield - Streamlit App
# Run: streamlit run streamlit_app.py
# Install: pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib plotly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudShield — Transaction Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0b0f; color: #e8eaf0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12141a;
        border-right: 1px solid #252836;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #12141a;
        border: 1px solid #252836;
        border-radius: 10px;
        padding: 12px;
    }

    /* Headers */
    h1, h2, h3 { color: #e8eaf0 !important; }
    h1 span { color: #ff4d6d; }

    /* Verdict box */
    .verdict-fraud {
        background: rgba(255,77,109,0.1);
        border: 2px solid #ff4d6d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #ff4d6d;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 0.08em;
    }
    .verdict-safe {
        background: rgba(0,212,170,0.1);
        border: 2px solid #00d4aa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #00d4aa;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 0.08em;
    }
    .verdict-review {
        background: rgba(255,179,64,0.1);
        border: 2px solid #ffb340;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #ffb340;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 0.08em;
    }

    /* Factor items */
    .factor-bad  { color: #ff4d6d; }
    .factor-warn { color: #ffb340; }
    .factor-good { color: #00d4aa; }

    /* Buttons */
    .stButton > button {
        background: #ff4d6d;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 0.06em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #ff2244;
        box-shadow: 0 0 16px rgba(255,77,109,0.5);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #12141a;
        border-bottom: 1px solid #252836;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6b7080;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: #ff4d6d !important;
        border-bottom: 2px solid #ff4d6d !important;
    }

    /* Info box */
    .info-box {
        background: rgba(0,212,170,0.05);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #8892a0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


@st.cache_resource
def train_model(filepath):
    """Train and cache the full ML pipeline."""
    df = pd.read_csv(filepath)

    # Feature engineering
    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Drop leaky/non-informative columns
    if "step" in df.columns:
        df.drop(columns="step", inplace=True)

    df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

    categorical = ["type"]
    numeric     = ["amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"]

    y = df_model["isFraud"]
    X = df_model.drop("isFraud", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "cm":        confusion_matrix(y_test, y_pred),
        "X_test":    X_test,
        "y_test":    y_test,
        "y_pred":    y_pred,
    }

    return pipeline, metrics


def predict_transaction(pipeline, tx_type, amount,
                         old_bal_org, new_bal_orig,
                         old_bal_dest, new_bal_dest):
    """Run fraud prediction on a single transaction."""
    bal_diff_orig = old_bal_org  - new_bal_orig
    bal_diff_dest = new_bal_dest - old_bal_dest

    sample = pd.DataFrame([{
        "type":            tx_type,
        "amount":          amount,
        "oldbalanceOrg":   old_bal_org,
        "newbalanceOrig":  new_bal_orig,
        "oldbalanceDest":  old_bal_dest,
        "newbalanceDest":  new_bal_dest,
        "balanceDiffOrig": bal_diff_orig,
        "balanceDiffDest": bal_diff_dest,
    }])

    prediction  = pipeline.predict(sample)[0]
    probability = pipeline.predict_proba(sample)[0][1]

    # Rule-based risk factors (from EDA)
    factors = []
    if tx_type in ["TRANSFER", "CASH_OUT"]:
        factors.append(("🔴", f"{tx_type} type — highest historical fraud rate"))
    else:
        factors.append(("🟢", f"{tx_type} type — near-zero fraud historically"))

    if old_bal_org > 0 and new_bal_orig == 0 and tx_type in ["TRANSFER", "CASH_OUT"]:
        factors.append(("🔴", "Account completely drained to zero — strong fraud signal"))

    if amount > old_bal_org * 1.5 and old_bal_org > 0:
        factors.append(("🟡", "Transaction amount greatly exceeds sender's prior balance"))

    if bal_diff_dest < 0:
        factors.append(("🔴", f"Receiver balance DECREASED by ₹{abs(bal_diff_dest):,.0f} — suspicious"))

    if probability < 0.25:
        verdict, css_class = "✅  LEGITIMATE",    "verdict-safe"
    elif probability < 0.55:
        verdict, css_class = "⚠️  REVIEW REQUIRED", "verdict-review"
    else:
        verdict, css_class = "🚨  FRAUD ALERT",   "verdict-fraud"

    return {
        "prediction":  prediction,
        "probability": round(probability * 100, 2),
        "verdict":     verdict,
        "css_class":   css_class,
        "factors":     factors,
    }


def make_gauge(probability):
    """Plotly gauge chart for fraud probability."""
    pct = probability

    if pct < 25:
        color = "#00d4aa"
    elif pct < 55:
        color = "#ffb340"
    else:
        color = "#ff4d6d"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        number= {"suffix": "%", "font": {"color": color, "size": 40}},
        title = {"text": "Fraud Probability", "font": {"color": "#6b7080", "size": 14}},
        gauge = {
            "axis": {
                "range": [0, 100],
                "tickcolor": "#252836",
                "tickfont": {"color": "#6b7080"},
            },
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#12141a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   25],  "color": "rgba(0,212,170,0.08)"},
                {"range": [25,  55],  "color": "rgba(255,179,64,0.08)"},
                {"range": [55,  100], "color": "rgba(255,77,109,0.08)"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="#0a0b0f",
        plot_bgcolor ="#0a0b0f",
        font_color   ="#e8eaf0",
        height       =260,
        margin       =dict(t=40, b=0, l=20, r=20),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("---")

    st.markdown("### 📁 Dataset")
    uploaded = st.file_uploader(
        "Upload AIML Dataset.csv",
        type=["csv"],
        help="Upload your PaySim transaction dataset"
    )

    # Save uploaded file temporarily
    dataset_path = None
    if uploaded:
        temp_path = "uploaded_dataset.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        dataset_path = temp_path
        st.success(f"✅ Dataset loaded")
    elif os.path.exists("AIML Dataset.csv"):
        dataset_path = "AIML Dataset.csv"
        st.info("Using 'AIML Dataset.csv' from directory")

    st.markdown("---")

    # Quick-fill presets
    st.markdown("### ⚡ Quick Presets")
    preset = st.radio(
        "Load example transaction:",
        ["Normal Payment", "Suspicious TRANSFER ⚠️", "Account Drain CASH_OUT ⚠️"],
        index=0,
    )

    PRESETS = {
        "Normal Payment": dict(
            tx_type="PAYMENT", amount=9839.64,
            old_bal_org=170136.0, new_bal_orig=160296.36,
            old_bal_dest=0.0, new_bal_dest=0.0,
        ),
        "Suspicious TRANSFER ⚠️": dict(
            tx_type="TRANSFER", amount=181000.0,
            old_bal_org=181000.0, new_bal_orig=0.0,
            old_bal_dest=0.0, new_bal_dest=0.0,
        ),
        "Account Drain CASH_OUT ⚠️": dict(
            tx_type="CASH_OUT", amount=229133.94,
            old_bal_org=15325.0, new_bal_orig=0.0,
            old_bal_dest=5083.0, new_bal_dest=51513.44,
        ),
    }

    selected = PRESETS[preset]

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b style='color:#00d4aa'>Model Info</b><br>
    Logistic Regression · balanced class weights<br>
    StandardScaler + OneHotEncoder(drop='first')<br>
    Trained on 4.45M transactions<br>
    Recall: 93% · Accuracy: 94.6%
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT — TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 EDA & Insights", "📈 Model Performance"])


# ============================================================
# TAB 1: PREDICT
# ============================================================

with tab1:
    st.markdown("## Transaction Risk Analyzer")

    # Dataset-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size",   "6.36M",  help="Total transactions in training set")
    col2.metric("Fraud Rate",     "0.13%",  help="Fraction of fraudulent transactions")
    col3.metric("Model Recall",   "93%",    help="Of actual frauds, how many were caught")
    col4.metric("Accuracy",       "94.6%",  help="Overall classification accuracy")

    st.markdown("---")

    left, right = st.columns([1.1, 0.9], gap="large")

    # --- INPUT FORM ---
    with left:
        st.markdown("### 📝 Transaction Details")

        tx_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
            index=["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"].index(selected["tx_type"]),
        )

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0, step=100.0,
            value=float(selected["amount"]),
            format="%.2f",
        )

        c1, c2 = st.columns(2)
        with c1:
            old_bal_org = st.number_input(
                "Sender Old Balance", min_value=0.0, step=100.0,
                value=float(selected["old_bal_org"]), format="%.2f",
            )
        with c2:
            new_bal_orig = st.number_input(
                "Sender New Balance", min_value=0.0, step=100.0,
                value=float(selected["new_bal_orig"]), format="%.2f",
            )

        c3, c4 = st.columns(2)
        with c3:
            old_bal_dest = st.number_input(
                "Receiver Old Balance", min_value=0.0, step=100.0,
                value=float(selected["old_bal_dest"]), format="%.2f",
            )
        with c4:
            new_bal_dest = st.number_input(
                "Receiver New Balance", min_value=0.0, step=100.0,
                value=float(selected["new_bal_dest"]), format="%.2f",
            )

        analyze_btn = st.button("▶  ANALYZE TRANSACTION", use_container_width=True)

    # --- RESULT PANEL ---
    with right:
        st.markdown("### 🎯 Risk Assessment")

        if analyze_btn or "result" not in st.session_state:
            if dataset_path:
                with st.spinner("Loading model..."):
                    pipeline, _ = train_model(dataset_path)
                result = predict_transaction(
                    pipeline, tx_type, amount,
                    old_bal_org, new_bal_orig,
                    old_bal_dest, new_bal_dest,
                )
                st.session_state["result"] = result
            else:
                st.warning("⚠️ Please upload your dataset in the sidebar to enable predictions.")
                st.stop()

        result = st.session_state.get("result")

        if result:
            # Gauge
            st.plotly_chart(
                make_gauge(result["probability"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Verdict badge
            st.markdown(
                f"<div class='{result['css_class']}'>{result['verdict']}</div>",
                unsafe_allow_html=True,
            )

            # Risk factors
            st.markdown("#### Risk Factors")
            for icon, text in result["factors"]:
                st.markdown(f"{icon} {text}")


# ============================================================
# TAB 2: EDA & INSIGHTS
# ============================================================

with tab2:
    st.markdown("## Exploratory Data Analysis")

    if not dataset_path:
        st.warning("⚠️ Upload a dataset in the sidebar to view EDA.")
        st.stop()

    with st.spinner("Loading dataset for EDA..."):
        df = load_data(dataset_path)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",    f"{df.shape[0]:,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Fraud Cases",   f"{df['isFraud'].sum():,}")
    col4.metric("Missing Values",df.isnull().sum().sum())

    st.markdown("---")

    row1_c1, row1_c2 = st.columns(2)

    # --- Plot 1: Transaction type distribution ---
    with row1_c1:
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig1 = px.bar(
            type_counts, x="Type", y="Count",
            title="Transaction Type Distribution",
            color="Count", color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig1.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a",
                           showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    # --- Plot 2: Fraud rate by type ---
    with row1_c2:
        fraud_rate = df.groupby("type")["isFraud"].mean().reset_index()
        fraud_rate.columns = ["Type", "Fraud Rate"]
        fraud_rate = fraud_rate.sort_values("Fraud Rate", ascending=False)
        fig2 = px.bar(
            fraud_rate, x="Type", y="Fraud Rate",
            title="Fraud Rate by Transaction Type",
            color="Fraud Rate", color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig2.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    # --- Plot 3: Amount distribution ---
    with row2_c1:
        log_amounts = np.log1p(df["amount"])
        fig3 = px.histogram(
            log_amounts, nbins=100,
            title="Transaction Amount Distribution (log scale)",
            labels={"value": "Log(Amount + 1)", "count": "Frequency"},
            template="plotly_dark", color_discrete_sequence=["#00d4aa"],
        )
        fig3.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a",
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # --- Plot 4: Frauds over time ---
    with row2_c2:
        if "step" in df.columns:
            frauds_t = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
            fig4 = px.line(
                x=frauds_t.index, y=frauds_t.values,
                title="Frauds Over Time (by Step)",
                labels={"x": "Step (Time)", "y": "Number of Frauds"},
                template="plotly_dark", color_discrete_sequence=["#ff4d6d"],
            )
            fig4.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Step column not available (was dropped during feature engineering).")

    # --- Fraud in TRANSFER vs CASH_OUT ---
    st.markdown("### TRANSFER & CASH_OUT Fraud Breakdown")
    fraud_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    counts = fraud_types.groupby(["type", "isFraud"]).size().reset_index(name="count")
    counts["isFraud"] = counts["isFraud"].map({0: "Genuine", 1: "Fraud"})
    fig5 = px.bar(
        counts, x="type", y="count", color="isFraud",
        barmode="group",
        title="Fraud Distribution in TRANSFER and CASH_OUT",
        color_discrete_map={"Genuine": "#00d4aa", "Fraud": "#ff4d6d"},
        template="plotly_dark",
    )
    fig5.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a")
    st.plotly_chart(fig5, use_container_width=True)

    # Raw data sample
    with st.expander("📋 Show Raw Data Sample (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with tab3:
    st.markdown("## Model Performance")

    if not dataset_path:
        st.warning("⚠️ Upload a dataset in the sidebar to view model metrics.")
        st.stop()

    with st.spinner("Training model and computing metrics..."):
        pipeline, metrics = train_model(dataset_path)

    # Key metrics
    report = metrics["report"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{metrics['accuracy']}%")
    col2.metric("Precision (Fraud)", f"{report['1']['precision']*100:.1f}%")
    col3.metric("Recall (Fraud)",    f"{report['1']['recall']*100:.1f}%")
    col4.metric("F1-Score (Fraud)",  f"{report['1']['f1-score']*100:.1f}%")

    st.markdown("---")

    left, right = st.columns(2)

    # --- Confusion Matrix ---
    with left:
        st.markdown("### Confusion Matrix")
        cm = metrics["cm"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Genuine", "Fraud"],
            y=["Genuine", "Fraud"],
            text_auto=True,
            color_continuous_scale="Reds",
            title="Confusion Matrix",
            template="plotly_dark",
        )
        fig_cm.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#12141a")
        st.plotly_chart(fig_cm, use_container_width=True)

    # --- Class report ---
    with right:
        st.markdown("### Classification Report")
        class_data = {
            "Class":     ["Genuine (0)", "Fraud (1)", "Macro Avg", "Weighted Avg"],
            "Precision": [
                f"{report['0']['precision']:.3f}",
                f"{report['1']['precision']:.3f}",
                f"{report['macro avg']['precision']:.3f}",
                f"{report['weighted avg']['precision']:.3f}",
            ],
            "Recall": [
                f"{report['0']['recall']:.3f}",
                f"{report['1']['recall']:.3f}",
                f"{report['macro avg']['recall']:.3f}",
                f"{report['weighted avg']['recall']:.3f}",
            ],
            "F1-Score": [
                f"{report['0']['f1-score']:.3f}",
                f"{report['1']['f1-score']:.3f}",
                f"{report['macro avg']['f1-score']:.3f}",
                f"{report['weighted avg']['f1-score']:.3f}",
            ],
            "Support": [
                f"{int(report['0']['support']):,}",
                f"{int(report['1']['support']):,}",
                f"{int(report['macro avg']['support']):,}",
                f"{int(report['weighted avg']['support']):,}",
            ],
        }
        st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Pipeline Architecture")
        st.code("""
Pipeline([
  ('prep', ColumnTransformer([
    ('num', StandardScaler(),
     ['amount','oldbalanceOrg','newbalanceOrig',
      'oldbalanceDest','newbalanceDest']),
    ('cat', OneHotEncoder(drop='first'), ['type'])
  ])),
  ('clf', LogisticRegression(
    class_weight='balanced',
    max_iter=1000
  ))
])
        """, language="python")

    # Save model button
    st.markdown("---")
    if st.button("💾 Save Model to Disk", use_container_width=False):
        joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
        st.success("✅ Model saved as `fraud_detection_pipeline.pkl`")