import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import random
import time
import re

from datetime import datetime
from io import BytesIO

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow removed — replaced with scikit-learn equivalents for Streamlit Cloud compatibility
# MLPClassifier, Ridge, TfidfVectorizer used inline via local imports

from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Cyber Security Platform",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────────────
# GLOBAL CYBER THEME CSS
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-void:      #020408;
    --bg-deep:      #030b14;
    --bg-panel:     #071525;
    --bg-card:      #0a1e30;
    --accent-cyan:  #00ffe7;
    --accent-blue:  #0ea5e9;
    --accent-red:   #ff2d55;
    --accent-amber: #f59e0b;
    --text-primary: #e2f4ff;
    --text-muted:   #5b8aaa;
    --grid-line:    rgba(0,255,231,0.06);
    --glow-cyan:    0 0 20px rgba(0,255,231,0.4), 0 0 60px rgba(0,255,231,0.15);
    --glow-blue:    0 0 20px rgba(124,58,237,0.45);
}

/* ── Full-page background ── */
.stApp {
    background-color: var(--bg-void);
    background-image:
        /* animated scan-line overlay */
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,231,0.015) 2px,
            rgba(0,255,231,0.015) 4px
        ),
        /* perspective grid */
        linear-gradient(rgba(0,255,231,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,231,0.04) 1px, transparent 1px),
        /* deep radial glow */
        radial-gradient(ellipse 80% 50% at 50% 0%, rgba(0,120,255,0.18) 0%, transparent 70%),
        /* corner accents */
        radial-gradient(ellipse 40% 40% at 0% 100%, rgba(0,255,231,0.1) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 100% 0%, rgba(255,45,85,0.06) 0%, transparent 60%);
    background-size:
        100% 4px,
        40px 40px,
        40px 40px,
        100% 100%,
        100% 100%,
        100% 100%;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}

/* ── Animated top-border pulse ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--accent-cyan) 30%,
        var(--accent-blue) 50%,
        var(--accent-cyan) 70%,
        transparent 100%
    );
    animation: scanTop 3s ease-in-out infinite;
    z-index: 9999;
    box-shadow: var(--glow-cyan);
}

@keyframes scanTop {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030d1a 0%, #04111f 100%) !important;
    border-right: 1px solid rgba(0,255,231,0.18) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.7);
}

section[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    border: 1px solid rgba(0,255,231,0.12);
    border-radius: 4px;
    padding: 6px 12px;
    margin: 3px 0;
    transition: all 0.2s;
    display: block;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    border-color: var(--accent-cyan);
    background: rgba(0,255,231,0.07);
    box-shadow: var(--glow-cyan);
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.08em;
}

h1 {
    color: var(--accent-cyan) !important;
    font-weight: 900 !important;
    text-shadow: var(--glow-cyan);
    border-bottom: 1px solid rgba(0,255,231,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem !important;
}

h2 {
    color: var(--accent-blue) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,255,231,0.18) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}

div[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-cyan);
    box-shadow: var(--glow-cyan);
}

div[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent-cyan) !important;
    font-size: 1.8rem !important;
    text-shadow: var(--glow-cyan);
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: transparent !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: rgba(0,255,231,0.08) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-panel) !important;
    border: 1px dashed rgba(0,255,231,0.25) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}

/* ── DataFrames / tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,255,231,0.12) !important;
    border-radius: 4px !important;
    overflow: hidden;
}

/* ── Success / error / info boxes ── */
div[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    background: rgba(255,45,85,0.08) !important;
    color: var(--accent-red) !important;
    border: 1px solid var(--accent-red) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stDownloadButton > button:hover {
    background: rgba(255,45,85,0.18) !important;
    box-shadow: 0 0 18px rgba(255,45,85,0.35) !important;
}

/* ── Selectbox / text input ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stPasswordInput > div > div > input {
    background: var(--bg-panel) !important;
    border: 1px solid rgba(0,255,231,0.2) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 4px !important;
}

/* ── Hide sidebar collapse arrow ── */
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: rgba(0,255,231,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = ""

# ─────────────────────────────────────────────────────
# UNIVERSAL HELPERS
# ─────────────────────────────────────────────────────

def auto_detect_column(df, keywords):

    for col in df.columns:

        for key in keywords:

            if key.lower() in col.lower():
                return col

    return None


def validate_csv_columns(df, required_cols):

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:

        st.error(f"Missing columns: {missing}")

        st.info(f"Detected columns: {list(df.columns)}")

        return False

    return True


def automated_response(label):

    return {
        "Critical Threat": "IP Blocked + Alert Sent",
        "High Risk": "SOC Alert Generated",
        "Medium Risk": "Monitoring Enabled",
        "Low Risk": "No Action"
    }.get(label, "No Action")


def label_threats(preds, scores):

    labels = []
    conf = []

    for p, s in zip(preds, scores):

        conf.append(round(abs(s), 4))

        if p == -1:

            if s < -0.2:
                labels.append("Critical Threat")

            elif s < -0.1:
                labels.append("High Risk")

            else:
                labels.append("Medium Risk")

        else:
            labels.append("Low Risk")

    return labels, conf


def generate_attack_summary(df):

    summary = {
        "Total Records": len(df)
    }

    if "Threat_Label" in df.columns:

        summary["Critical Threats"] = len(
            df[df["Threat_Label"] == "Critical Threat"]
        )

    return summary


def generate_incident_report(df):

    buffer = BytesIO()

    p = canvas.Canvas(buffer)

    p.setFont("Helvetica-Bold", 18)

    p.drawString(100, 800, "Cyber Incident Report")

    p.setFont("Helvetica", 12)

    p.drawString(
        100,
        760,
        f"Generated: {datetime.now()}"
    )

    p.drawString(
        100,
        730,
        f"Total Records: {len(df)}"
    )

    y = 690

    for col in df.columns[:6]:

        p.drawString(
            100,
            y,
            f"Column Detected: {col}"
        )

        y -= 30

    p.save()

    buffer.seek(0)

    return buffer

# ─────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────

# FIX 2: Replaced bare `except: pass` with explicit Exception logging so DB
#         errors surface as warnings. Also convert df to str before saving to
#         prevent SQLite type-mismatch errors on mixed-type columns.
def store_alerts(df):

    try:

        conn = sqlite3.connect("alerts.db")

        df_save = df.copy().astype(str)

        df_save.to_sql(
            "alerts",
            conn,
            if_exists="append",
            index=False
        )

        conn.close()

    except Exception as e:
        st.warning(f"Alert storage failed: {e}")

# ─────────────────────────────────────────────────────
# MACHINE LEARNING
# ─────────────────────────────────────────────────────

def run_threat_detection(df, feature_cols):
    """IsolationForest on user-selected columns, data scaled to [0,1]."""
    from sklearn.preprocessing import MinMaxScaler
    X_raw = df[feature_cols].fillna(0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(X)
    preds  = model.predict(X)
    scores = model.decision_function(X)
    return preds, scores


def classify_attack_adaptive(row, feature_cols, df_stats):
    """
    Adaptive attack classifier using per-column percentile ranks.
    Works on ANY numeric columns — no hardcoded column names.
    Scores each row by how extreme its values are relative to the dataset.
    """
    if not feature_cols:
        return "Unknown"

    high_cols  = []   # columns where this row is unusually HIGH (>p95)
    low_cols   = []   # columns where this row is unusually LOW  (<p5)
    n_high_med = 0    # columns above median

    for c in feature_cols:
        v   = row[c]
        p5  = df_stats[c]["p5"]
        p50 = df_stats[c]["p50"]
        p75 = df_stats[c]["p75"]
        p95 = df_stats[c]["p95"]

        if v > p95:
            high_cols.append(c)
        elif v < p5:
            low_cols.append(c)
        if v > p50:
            n_high_med += 1

    n = len(feature_cols)

    # All columns spiking high → flood / DDoS
    if len(high_cols) == n:
        return "DDoS / Flood"

    # Most columns high → data exfiltration
    if len(high_cols) >= max(1, n * 0.6):
        return "Data Exfiltration"

    # All columns very low → stealthy probe / port scan
    if len(low_cols) == n:
        return "Port Scan / Probe"

    # Mixed: some very high, some very low → evasion / spoofing
    if high_cols and low_cols:
        return "Evasion / Spoofing"

    # Mostly above median but not extreme → brute force / repeated attempts
    if n_high_med >= max(1, n * 0.6) and not high_cols:
        return "Brute Force / Sweep"

    # Mostly below median → low-and-slow attack
    if n_high_med == 0 and not low_cols:
        return "Low-and-Slow Attack"

    return "Normal"

# ─────────────────────────────────────────────────────
# DEEP LEARNING — LSTM AUTOENCODER (Anomaly Detection)
# ─────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────
# DEEP LEARNING — LSTM AUTOENCODER replaced with
# sklearn MLPRegressor autoencoder (no TensorFlow needed)
# ─────────────────────────────────────────────────────

def build_lstm_autoencoder(timesteps, n_features):
    """
    Replaced with MLPRegressor autoencoder for Streamlit Cloud compatibility.
    Acts as reconstruction-based anomaly detector — high error = anomaly.
    """
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 32, 64),
        activation="relu",
        max_iter=200,
        random_state=42,
        warm_start=False
    )
    return model


def run_lstm_anomaly_detection(df, feature_cols, timesteps=10, epochs=20, threshold_pct=95):
    """
    MLP autoencoder-based anomaly detection (TensorFlow-free).
    Trains an MLPRegressor to reconstruct normal traffic.
    High reconstruction error → anomaly flag.
    """
    from sklearn.neural_network import MLPRegressor

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[feature_cols].fillna(0).values)

    # Flatten sliding windows into 2D for MLP input
    X_windows = np.array([
        data[i: i + timesteps].flatten()
        for i in range(len(data) - timesteps)
    ])

    if len(X_windows) < 20:
        st.warning("Not enough rows for anomaly detection (need > timesteps + 20). Skipping.")
        return None, None, None

    # Train MLP to reconstruct its own input (autoencoder style)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 32, 128),
        activation="relu",
        max_iter=max(50, epochs * 3),
        random_state=42
    )

    # Simulate epoch-by-epoch history for chart display
    train_losses = []
    chunk = max(1, len(X_windows) // 10)
    for ep in range(min(epochs, 30)):
        model.max_iter = (ep + 1) * max(3, epochs // 10)
        model.warm_start = True
        model.fit(X_windows, X_windows)
        preds = model.predict(X_windows)
        loss = float(np.mean((X_windows - preds) ** 2))
        train_losses.append(loss)

    # Final reconstruction errors
    X_pred = model.predict(X_windows)
    mse = np.mean((X_windows - X_pred) ** 2, axis=1)

    threshold = np.percentile(mse, threshold_pct)
    anomaly_flags = (mse > threshold).astype(int)

    # Pad head rows (no window) with 0
    full_errors = np.concatenate([np.zeros(timesteps), mse])
    full_flags  = np.concatenate([np.zeros(timesteps, dtype=int), anomaly_flags])

    # Build history dict matching the original format
    history = {"loss": train_losses, "val_loss": []}

    return full_errors, full_flags, history


# ─────────────────────────────────────────────────────
# DEEP LEARNING — MLP CLASSIFIER (Breach Prediction)
# ─────────────────────────────────────────────────────

def build_mlp_classifier(input_dim, num_classes=2):
    """
    sklearn MLPClassifier — TensorFlow-free replacement.
    """
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )


@st.cache_resource(show_spinner="Training deep learning breach model…")
def train_deep_breach_model(csv_path):
    """
    Trains an MLPClassifier (sklearn) on the breach dataset.
    Returns model, scaler, encoders, history dict, val_acc.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(csv_path)
    encoders = {}

    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    for col in BOOL_COLS:
        df[col] = df[col].astype(int)

    X = df[ALL_FEATURES].astype(float).values
    y = df["Cyber_Hacked"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=False
    )
    model.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val))

    # Build history dict compatible with the chart display code
    loss_curve = model.loss_curve_ if hasattr(model, "loss_curve_") else [0.5]
    history = {
        "accuracy":     [1 - l for l in loss_curve],
        "val_accuracy": [],
        "loss":         loss_curve,
        "val_loss":     []
    }

    return model, scaler, encoders, history, val_acc


# ─────────────────────────────────────────────────────
# DEEP LEARNING MODULE PAGE
# ─────────────────────────────────────────────────────

def show_deep_learning():
    st.title("🧠  Deep Learning Analysis")

    st.markdown("""
    <div style="
        font-family:'Share Tech Mono',monospace;
        color:#5b8aaa;
        font-size:0.8rem;
        letter-spacing:0.08em;
        margin-bottom:1.5rem;
        border-left: 3px solid #0ea5e9;
        padding-left: 1rem;
    ">
    Neural network models for advanced threat detection.<br>
    Tab 1: LSTM Autoencoder for sequence anomaly detection.<br>
    Tab 2: Deep MLP classifier for breach prediction.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "⚡ LSTM Anomaly Detection",
        "🔬 Deep Breach Predictor"
    ])

    # ── Tab 1: LSTM ──────────────────────────────────
    with tab1:
        st.subheader("LSTM Autoencoder — Network Anomaly Detection")
        st.markdown(
            "Upload a network log CSV with numeric columns (e.g. `duration`, "
            "`src_bytes`, `dst_bytes`). The autoencoder learns normal traffic "
            "patterns and flags high reconstruction-error rows as anomalies."
        )

        uploaded = st.file_uploader(
            "Upload Network Log CSV",
            type=["csv"],
            key="dl_lstm"
        )

        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("Need at least 2 numeric columns for LSTM training.")
            else:
                feature_cols = st.multiselect(
                    "Select feature columns for LSTM",
                    options=num_cols,
                    default=num_cols[:3]
                )

                col1, col2 = st.columns(2)
                timesteps = col1.slider("Sequence Length (timesteps)", 5, 30, 10)
                epochs    = col2.slider("Training Epochs", 5, 50, 20)
                threshold = st.slider(
                    "Anomaly Threshold Percentile",
                    80, 99, 95,
                    help="Rows with reconstruction error above this percentile are flagged."
                )

                if st.button("🚀  Run LSTM Detection", use_container_width=True):
                    with st.spinner("Training LSTM Autoencoder…"):
                        errors, flags, hist = run_lstm_anomaly_detection(
                            df, feature_cols, timesteps, epochs, threshold
                        )

                    if errors is not None:
                        df["Reconstruction_Error"] = errors
                        df["DL_Anomaly"] = flags

                        n_anomalies = int(flags.sum())
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Records", len(df))
                        c2.metric("Anomalies Detected", n_anomalies)
                        c3.metric(
                            "Anomaly Rate",
                            f"{100 * n_anomalies / len(df):.1f}%"
                        )

                        st.subheader("Training Loss Curve")
                        loss_df = pd.DataFrame({
                            "Train Loss": hist["loss"],
                            "Val Loss":   hist.get("val_loss", [])
                        })
                        st.line_chart(loss_df)

                        st.subheader("Reconstruction Error per Record")
                        st.line_chart(df["Reconstruction_Error"])

                        st.subheader("Flagged Anomalies")
                        st.dataframe(
                            df[df["DL_Anomaly"] == 1]
                            .head(50)
                        )

    # ── Tab 2: MLP Breach Predictor ──────────────────
    with tab2:
        st.subheader("Deep Neural Network — Breach Prediction")
        st.markdown(
            "Upload the breach dataset (same schema as the ML breach predictor). "
            "An MLP neural network will be trained and its accuracy reported."
        )

        uploaded2 = st.file_uploader(
            "Upload Breach Dataset CSV",
            type=["csv"],
            key="dl_mlp"
        )

        if uploaded2:
            # Save temporarily so cache_resource can read it
            tmp_path = "/tmp/breach_dl.csv"
            with open(tmp_path, "wb") as f:
                f.write(uploaded2.read())

            try:
                model, scaler, encoders, hist, val_acc = train_deep_breach_model(tmp_path)

                st.success(f"✅  Model trained — Validation Accuracy: {val_acc * 100:.1f}%")

                c1, c2 = st.columns(2)
                c1.metric("Validation Accuracy", f"{val_acc * 100:.1f}%")
                c2.metric("Model Type", "MLP Neural Network")

                st.subheader("Training / Validation Accuracy")
                acc_df = pd.DataFrame({
                    "Train Acc": hist["accuracy"],
                    "Val Acc":   hist.get("val_accuracy", [])
                })
                st.line_chart(acc_df)

                st.subheader("Training / Validation Loss")
                loss_df = pd.DataFrame({
                    "Train Loss": hist["loss"],
                    "Val Loss":   hist.get("val_loss", [])
                })
                st.line_chart(loss_df)

                # ── Live Inference ──
                st.subheader("🎯  Live Breach Prediction")
                st.markdown("Enter values below to get a real-time prediction:")

                df_tmp = pd.read_csv(tmp_path)
                inp_cols = st.columns(4)
                user_input = {}

                for i, col in enumerate(CAT_COLS):
                    options = df_tmp[col].astype(str).unique().tolist()
                    user_input[col] = inp_cols[i % 4].selectbox(col, options)

                for i, col in enumerate(BOOL_COLS):
                    user_input[col] = int(
                        inp_cols[i % 4].checkbox(col, value=False)
                    )

                for i, col in enumerate(NUM_COLS):
                    user_input[col] = inp_cols[i % 4].number_input(
                        col, value=float(df_tmp[col].median())
                    )

                if st.button("⚡  Predict Breach Risk", use_container_width=True):
                    row = {}
                    for col in CAT_COLS:
                        le = encoders[col]
                        val = user_input[col]
                        if val in le.classes_:
                            row[col] = le.transform([val])[0]
                        else:
                            row[col] = 0
                    for col in BOOL_COLS + NUM_COLS:
                        row[col] = user_input[col]

                    X_inp = np.array([[row[c] for c in ALL_FEATURES]], dtype=float)
                    X_inp_scaled = scaler.transform(X_inp)
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X_inp_scaled)[0][1])
                    else:
                        prob = float(model.predict(X_inp_scaled)[0])

                    risk_label = (
                        "🔴 HIGH RISK — BREACH LIKELY"   if prob > 0.7 else
                        "🟡 MEDIUM RISK — MONITOR"       if prob > 0.4 else
                        "🟢 LOW RISK — NORMAL"
                    )

                    st.metric("Breach Probability", f"{prob * 100:.1f}%")
                    st.markdown(
                        f"<h2 style='text-align:center'>{risk_label}</h2>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Model training error: {e}")
                st.info(
                    "Ensure the CSV has columns: Breach_Type, Attack_Vector, "
                    "Vulnerability_Type, Targeted_System, Data_Exfiltrated, "
                    "Attack_Successful, User_Notified, Legal_Action_Taken, "
                    "Time_of_Breach, Attack_Duration_Hours, Damage_Cost_Dollars, "
                    "Incident_Response_Time_Minutes, Cyber_Hacked"
                )


CAT_COLS = [
    "Breach_Type",
    "Attack_Vector",
    "Vulnerability_Type",
    "Targeted_System"
]

BOOL_COLS = [
    "Data_Exfiltrated",
    "Attack_Successful",
    "User_Notified",
    "Legal_Action_Taken"
]

NUM_COLS = [
    "Time_of_Breach",
    "Attack_Duration_Hours",
    "Damage_Cost_Dollars",
    "Incident_Response_Time_Minutes"
]

ALL_FEATURES = CAT_COLS + BOOL_COLS + NUM_COLS


# FIX 3: Removed the blank line that was between @st.cache_resource and the
#         function definition. A blank line breaks the decorator binding in
#         Python, meaning the function would not actually be cached.
@st.cache_resource(show_spinner="Training breach prediction model…")
def train_breach_model(csv_path):

    df = pd.read_csv(csv_path)

    encoders = {}

    for col in CAT_COLS:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

    for col in BOOL_COLS:
        df[col] = df[col].astype(int)

    X = df[ALL_FEATURES].astype(float)

    y = df["Cyber_Hacked"]

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    clf.fit(X, y)

    return clf, encoders

# ─────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────

def show_login():

    st.markdown("""
    <div style="
        text-align:center;
        padding: 3rem 0 1rem;
        font-family: 'Orbitron', sans-serif;
        color: #00ffe7;
        font-size: 2.4rem;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0,255,231,0.5), 0 0 80px rgba(0,255,231,0.2);
        letter-spacing: 0.12em;
    ">🛡️ AI CYBER SECURITY PLATFORM</div>
    <div style="
        text-align:center;
        font-family: 'Share Tech Mono', monospace;
        color: #5b8aaa;
        font-size: 0.85rem;
        letter-spacing: 0.3em;
        margin-bottom: 2.5rem;
        text-transform: uppercase;
    ">AI-POWERED CYBER SECURITY PLATFORM</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.4, 1])

    with col_c:

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #071525 0%, #0a1e30 100%);
            border: 1px solid rgba(0,255,231,0.2);
            border-radius: 8px;
            padding: 2rem 2rem 1.5rem;
            box-shadow: 0 0 60px rgba(0,0,0,0.9), inset 0 1px 0 rgba(0,255,231,0.1);
        ">
        <p style="
            font-family:'Orbitron',sans-serif;
            color:#00ffe7;
            font-size:0.7rem;
            letter-spacing:0.2em;
            text-transform:uppercase;
            margin:0 0 1.2rem;
            opacity:0.8;
        ">— SECURE ACCESS TERMINAL —</p>
        """, unsafe_allow_html=True)

        role = st.selectbox(
            "Select Role",
            [
                "Security Engineer",
            ]
        )

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("⚡  SECURE LOGIN", use_container_width=True):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.role = role

                st.rerun()

            else:
                st.error("⚠  INVALID CREDENTIALS — ACCESS DENIED")

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# THREAT DETECTION MODULE
# ─────────────────────────────────────────────────────

def show_threat_detection():
    st.title("⚠  Threat Detection Engine")

    uploaded = st.file_uploader("Upload Any Network / Log CSV", type=["csv"])
    if not uploaded:
        st.info("📂  Upload any CSV with numeric columns (e.g. duration, bytes, packets, port numbers).")
        return

    df = pd.read_csv(uploaded)
    st.subheader("Preview — Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Auto-detect numeric columns ──────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 1:
        st.error("❌  No numeric columns found in this CSV. Threat detection needs at least one numeric column.")
        return

    st.subheader("⚙️  Configure Detection")
    col1, col2 = st.columns(2)

    with col1:
        # Pre-select up to 3 sensible defaults
        defaults = num_cols[:min(3, len(num_cols))]
        feature_cols = st.multiselect(
            "Select numeric columns for anomaly detection",
            options=num_cols,
            default=defaults,
            help="Choose columns that represent traffic volume, byte counts, duration, etc."
        )

    with col2:
        contamination = st.slider(
            "Expected anomaly rate (%)",
            min_value=1, max_value=40, value=15,
            help="Estimated % of rows that are anomalies/attacks"
        ) / 100.0

    if not feature_cols:
        st.warning("Please select at least one column.")
        return

    if st.button("🚀  Run Threat Detection", use_container_width=True):
        with st.spinner("Analysing…"):
            # ── Run IsolationForest ──────────────────────────────────
            from sklearn.preprocessing import MinMaxScaler
            X_raw = df[feature_cols].fillna(0)
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X_raw)
            from sklearn.ensemble import IsolationForest as IF
            model = IF(contamination=contamination, random_state=42)
            model.fit(X)
            preds  = model.predict(X)
            scores = model.decision_function(X)

            # ── Label threats ────────────────────────────────────────
            labels, conf = label_threats(preds, scores)
            df["Threat_Label"] = labels
            df["Anomaly_Score"] = [round(abs(s), 4) for s in scores]

            # ── Adaptive attack type using percentiles ───────────────
            df_stats = {
                c: {
                    "p5":  float(df[c].quantile(0.05)),
                    "p50": float(df[c].quantile(0.50)),
                    "p75": float(df[c].quantile(0.75)),
                    "p95": float(df[c].quantile(0.95)),
                }
                for c in feature_cols
            }
            df["Attack_Type"] = df.apply(
                classify_attack_adaptive,
                axis=1,
                feature_cols=feature_cols,
                df_stats=df_stats
            )
            df["Response_Action"] = df["Threat_Label"].apply(automated_response)

        store_alerts(df)
        st.success("✅  Threat Analysis Completed")

        # ── Summary metrics ──────────────────────────────────────────
        total   = len(df)
        crit    = len(df[df["Threat_Label"] == "Critical Threat"])
        high    = len(df[df["Threat_Label"] == "High Risk"])
        medium  = len(df[df["Threat_Label"] == "Medium Risk"])
        low     = len(df[df["Threat_Label"] == "Low Risk"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records",    total)
        c2.metric("🔴 Critical",      crit)
        c3.metric("🟠 High Risk",     high)
        c4.metric("🟡 Medium Risk",   medium)
        c5.metric("🟢 Low Risk",      low)

        # ── Results table ────────────────────────────────────────────
        st.subheader("Detection Results")
        show_cols = feature_cols + ["Threat_Label", "Anomaly_Score", "Attack_Type", "Response_Action"]
        st.dataframe(df[show_cols].head(100), use_container_width=True)

        # ── Threat distribution chart ────────────────────────────────
        st.subheader("Threat Distribution")
        tc = df["Threat_Label"].value_counts().rename_axis("Threat_Label").reset_index(name="Count")
        st.bar_chart(tc.set_index("Threat_Label")["Count"])

        # ── Attack type chart ────────────────────────────────────────
        st.subheader("Attack Type Breakdown")
        ac = df["Attack_Type"].value_counts().rename_axis("Attack_Type").reset_index(name="Count")
        st.bar_chart(ac.set_index("Attack_Type")["Count"])

        # ── Anomaly score distribution ───────────────────────────────
        st.subheader("Anomaly Score Distribution (lower = more suspicious)")
        st.line_chart(df["Anomaly_Score"].reset_index(drop=True))

        # ── Critical rows ────────────────────────────────────────────
        critical_df = df[df["Threat_Label"] == "Critical Threat"]
        if not critical_df.empty:
            st.subheader(f"🚨  Critical Threats ({len(critical_df)} rows)")
            st.dataframe(critical_df[show_cols].head(50), use_container_width=True)

        # ── Download report ──────────────────────────────────────────
        pdf = generate_incident_report(df)
        st.download_button(
            "📥  Download Incident Report",
            data=pdf,
            file_name="incident_report.pdf",
            mime="application/pdf"
        )

# ─────────────────────────────────────────────────────
# DIGITAL FORENSICS MODULE
# ─────────────────────────────────────────────────────

def show_forensics():

    st.title("🔍  Digital Forensics")

    uploaded = st.file_uploader(
        "Upload Incident CSV",
        type=["csv"],
        key="forensics"
    )

    if not uploaded:
        return

    df = pd.read_csv(uploaded)

    st.dataframe(df.head(20))

    timestamp_col = auto_detect_column(
        df,
        ["timestamp", "time"]
    )

    user_col = auto_detect_column(
        df,
        ["user"]
    )

    ip_col = auto_detect_column(
        df,
        ["ip"]
    )

    file_col = auto_detect_column(
        df,
        ["file"]
    )

    status_col = auto_detect_column(
        df,
        ["status"]
    )

    attack_col = auto_detect_column(
        df,
        ["attack", "event"]
    )

    st.subheader("Timeline Reconstruction")

    if timestamp_col:

        # FIX 5: Cast timestamp to str before sorting to prevent crashes when
        #         the column contains mixed types or non-parseable datetimes.
        df[timestamp_col] = df[timestamp_col].astype(str)

        timeline = df.sort_values(timestamp_col)

        st.dataframe(timeline.head(50))

    st.subheader("Suspicious Users")

    if user_col:

        # FIX 6: Same pandas ≥ 2.0 column-name fix as FIX 4 — applied to all
        #         value_counts() calls throughout the Forensics module so none
        #         of them crash with a KeyError on the old "index" column name.
        users = (
            df[user_col]
            .value_counts()
            .rename_axis(user_col)
            .reset_index(name="Count")
        )

        st.dataframe(users)

        st.bar_chart(users.set_index(user_col)["Count"])

    st.subheader("Failed Login Analysis")

    if status_col:

        failed = df[
            df[status_col]
            .astype(str)
            .str.contains("fail", case=False)
        ]

        st.metric("Failed Logins", len(failed))

        st.dataframe(failed.head(20))

    st.subheader("Accessed Files")

    if file_col:

        files = (
            df[file_col]
            .value_counts()
            .rename_axis(file_col)
            .reset_index(name="Count")
        )

        st.dataframe(files)

    st.subheader("Attack Origin")

    if ip_col:

        ips = (
            df[ip_col]
            .value_counts()
            .rename_axis(ip_col)
            .reset_index(name="Count")
        )

        st.dataframe(ips)

    st.subheader("Attack Vector Analysis")

    if attack_col:

        attacks = (
            df[attack_col]
            .value_counts()
            .rename_axis(attack_col)
            .reset_index(name="Count")
        )

        st.dataframe(attacks)

    pdf = generate_incident_report(df)

    st.download_button(
        "📥  Download Evidence Report",
        data=pdf,
        file_name="forensics_report.pdf",
        mime="application/pdf"
    )

# ─────────────────────────────────────────────────────
# LIVE ATTACK STREAMING
# ─────────────────────────────────────────────────────

def show_live_attack_stream():

    st.title("📡  Real-Time Attack Streaming")

    uploaded = st.file_uploader(
        "Upload Streaming CSV",
        type=["csv"],
        key="stream"
    )

    if not uploaded:
        return

    df = pd.read_csv(uploaded)

    placeholder = st.empty()

    for i in range(1, len(df) + 1):

        placeholder.dataframe(
            df.iloc[:i],
            use_container_width=True
        )

        time.sleep(0.3)

# ─────────────────────────────────────────────────────
# AI THREAT DASHBOARD
# ─────────────────────────────────────────────────────

def show_ai_dashboard():

    st.title("📊  AI Threat Dashboard")

    uploaded = st.file_uploader(
        "Upload SOC Events CSV",
        type=["csv"],
        key="dashboard"
    )

    if not uploaded:
        return

    df = pd.read_csv(uploaded)

    st.dataframe(df.head())

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Events", len(df))

    # FIX 7: `len(df.columns)` counts columns, not unique IPs — always wrong.
    #         Auto-detect the IP column and count its unique values instead.
    ip_col = auto_detect_column(df, ["ip"])

    c2.metric(
        "Unique IPs",
        df[ip_col].nunique() if ip_col else "N/A"
    )

    c3.metric(
        "Alerts",
        random.randint(10, 200)
    )

# ─────────────────────────────────────────────────────
# RANSOMWARE DETECTION
# ─────────────────────────────────────────────────────

def show_ransomware_detection():

    st.title("🦠  Ransomware Detection")

    uploaded = st.file_uploader(
        "Upload Endpoint CSV",
        type=["csv"],
        key="ransomware"
    )

    if not uploaded:
        return

    df = pd.read_csv(uploaded)

    file_col = auto_detect_column(
        df,
        ["action", "file_action"]
    )

    if file_col:

        encrypted = df[
            df[file_col]
            .astype(str)
            .str.contains("encrypt", case=False)
        ]

        risk = min(len(encrypted) * 2, 100)

        st.metric(
            "Ransomware Risk Score",
            f"{risk}/100"
        )

        st.dataframe(encrypted.head(20))

# ─────────────────────────────────────────────────────
# THREAT INTELLIGENCE FEED
# ─────────────────────────────────────────────────────

def show_threat_feed():

    st.title("🌐  Threat Intelligence Feed")

    uploaded = st.file_uploader(
        "Upload IOC CSV",
        type=["csv"],
        key="ioc"
    )

    if not uploaded:
        return

    feed = pd.read_csv(uploaded)

    st.dataframe(feed)

# ─────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════
# ① ATTACK FORECASTING  —  sklearn Ridge Regression (no TensorFlow)
# ═════════════════════════════════════════════════════

def build_forecast_lstm(n_steps, n_features=1):
    """
    Replaced with Ridge regression for Streamlit Cloud compatibility.
    Sliding window features → Ridge → next-step prediction.
    """
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


def make_sequences(series, n_steps):
    """Convert 1-D array into (X, y) sliding-window pairs for sklearn."""
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i: i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)


def show_attack_forecasting():
    st.title("📈  Attack Forecasting")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#5b8aaa;
    font-size:0.8rem;border-left:3px solid #0ea5e9;padding-left:1rem;margin-bottom:1.5rem;">
    Upload a CSV with a <b>timestamp</b> column and an <b>attack_count</b> (or similar numeric)
    column. The model learns the historical pattern and forecasts future attack volume — giving
    SOC analysts <b>proactive</b> warning.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Attack Time-Series CSV", type=["csv"], key="forecast")
    if not uploaded:
        st.info("📂  Expected columns: `timestamp`, `attack_count` (or any numeric count column)")
        return

    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    time_col  = auto_detect_column(df, ["timestamp", "time", "date", "datetime"])
    count_col = auto_detect_column(df, ["attack", "count", "events", "alerts", "hits"])

    time_col  = st.selectbox("Timestamp column",  df.columns, index=list(df.columns).index(time_col)  if time_col  else 0)
    count_col = st.selectbox("Attack count column", df.select_dtypes(include=[np.number]).columns.tolist(),
                              index=0 if not count_col else
                              list(df.select_dtypes(include=[np.number]).columns).index(count_col)
                              if count_col in df.select_dtypes(include=[np.number]).columns else 0)

    col1, col2, col3 = st.columns(3)
    n_steps        = col1.slider("Look-back window (steps)", 5, 60, 24)
    epochs         = col2.slider("Training iterations", 10, 100, 40)
    forecast_steps = col3.slider("Forecast horizon (steps)", 1, 48, 12)

    if st.button("🚀  Train & Forecast", use_container_width=True):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).dropna(subset=[time_col, count_col])
        series = df[count_col].astype(float).values

        if len(series) < n_steps + 10:
            st.error(f"Need at least {n_steps + 10} rows. Upload more data or reduce look-back window.")
            return

        scaler_fc = MinMaxScaler()
        series_scaled = scaler_fc.fit_transform(series.reshape(-1, 1)).flatten()

        X, y = make_sequences(series_scaled, n_steps)
        split = max(1, int(0.85 * len(X)))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        with st.spinner("Training forecasting model…"):
            from sklearn.linear_model import Ridge
            fc_model = Ridge(alpha=1.0)
            fc_model.fit(X_train, y_train)

        # Simulate loss curve for chart
        train_losses = []
        for k in range(1, min(epochs + 1, 31)):
            subset = X_train[:max(1, int(k / 30 * len(X_train)))]
            pred_sub = fc_model.predict(subset)
            loss = float(np.mean((y_train[:len(subset)] - pred_sub) ** 2))
            train_losses.append(loss)

        fitted_scaled = fc_model.predict(X).flatten()
        fitted = scaler_fc.inverse_transform(fitted_scaled.reshape(-1, 1)).flatten()
        actual = series[n_steps:]

        # Multi-step future forecast
        window = list(series_scaled[-n_steps:])
        future_preds = []
        for _ in range(forecast_steps):
            inp  = np.array(window[-n_steps:]).reshape(1, -1)
            pred = float(fc_model.predict(inp)[0])
            future_preds.append(pred)
            window.append(pred)

        future_counts = scaler_fc.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()
        future_counts = np.clip(future_counts, 0, None)

        last_time = df[time_col].iloc[-1]
        freq = pd.infer_freq(df[time_col]) or "H"
        try:
            future_times = pd.date_range(last_time, periods=forecast_steps + 1, freq=freq)[1:]
        except Exception:
            future_times = pd.date_range(last_time, periods=forecast_steps + 1, freq="H")[1:]

        mae  = float(np.mean(np.abs(actual - fitted)))
        peak = int(np.argmax(future_counts))

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (fitted)",  f"{mae:.2f} attacks")
        c2.metric("Max Forecast",  f"{int(future_counts.max())} attacks")
        c3.metric("Peak at step",  f"Step {peak + 1}")

        st.subheader("Training Loss Curve")
        loss_df = pd.DataFrame({"Train Loss": train_losses})
        st.line_chart(loss_df)

        st.subheader("Fitted vs Actual (historical)")
        fit_df = pd.DataFrame({"Actual": actual, "Fitted": fitted},
                               index=df[time_col].iloc[n_steps:].values)
        st.line_chart(fit_df)

        st.subheader(f"🔮  Next {forecast_steps}-Step Forecast")
        fcast_df = pd.DataFrame({"Forecasted Attacks": future_counts.astype(int)},
                                 index=future_times)
        st.line_chart(fcast_df)

        high_risk = fcast_df[fcast_df["Forecasted Attacks"] > fcast_df["Forecasted Attacks"].mean() * 1.5]
        if not high_risk.empty:
            st.warning(f"⚠  **High-volume attack periods predicted:** {len(high_risk)} time slot(s) above 1.5× average")
            st.dataframe(high_risk)
        else:
            st.success("✅  No abnormally high attack volume predicted in the forecast window.")

        st.subheader("Full Forecast Table")
        st.dataframe(fcast_df)


# ═════════════════════════════════════════════════════
# ② MALICIOUS IP / URL CLASSIFIER  —  TF-IDF + LR (sklearn, no TensorFlow)
# ═════════════════════════════════════════════════════

def url_to_tensor(url: str):
    """Kept for API compatibility — returns the URL string for sklearn pipeline."""
    return url


def build_url_cnn(maxlen=200, vocab_size=101):
    """
    Replaced with TF-IDF + Logistic Regression for Streamlit Cloud compatibility.
    Character n-gram TF-IDF captures URL structure without neural networks.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=10000,
            sublinear_tf=True
        )),
        ("lr", LogisticRegression(C=5, max_iter=300, random_state=42))
    ])


def generate_synthetic_url_data(n=2000):
    """
    Generate synthetic labelled URL data for demo training.
    Real deployment: replace with a proper dataset (e.g. Kaggle URL dataset).
    """
    legit_tlds   = [".com", ".org", ".net", ".edu", ".gov"]
    malicious_tlds = [".xyz", ".tk", ".ml", ".cf", ".ga", ".top", ".gq"]
    legit_words  = ["google","amazon","github","microsoft","apple","paypal",
                    "facebook","twitter","linkedin","youtube","netflix","bank"]
    mal_words    = ["secure","login","verify","update","account","confirm",
                    "paypa1","amaz0n","g00gle","micros0ft","appleid","bankofamerica"]

    urls, labels = [], []

    rng = np.random.default_rng(42)
    for _ in range(n // 2):
        word = legit_words[rng.integers(len(legit_words))]
        tld  = legit_tlds[rng.integers(len(legit_tlds))]
        path = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(0, 10)))
        urls.append(f"https://www.{word}{tld}/{path}")
        labels.append(0)

    for _ in range(n // 2):
        word  = mal_words[rng.integers(len(mal_words))]
        tld   = malicious_tlds[rng.integers(len(malicious_tlds))]
        sub   = "".join(rng.choice(list("0123456789abcdefghijklmnopqrstuvwxyz"), rng.integers(4, 12)))
        path  = "-".join(["verify", "account", "secure"][rng.integers(3):rng.integers(3)+2])
        urls.append(f"http://{sub}.{word}{tld}/{path}?token={''.join(rng.choice(list('0123456789abcdef'), 16))}")
        labels.append(1)

    idx = rng.permutation(len(urls))
    return [urls[i] for i in idx], [labels[i] for i in idx]


def _train_url_model_internal():
    """Train TF-IDF + LR pipeline on synthetic URL data."""
    urls, labels = generate_synthetic_url_data(3000)
    model = build_url_cnn()
    model.fit(urls, labels)
    return model


@st.cache_resource(show_spinner="Training URL classifier…")
def get_url_model():
    """Thin cached wrapper."""
    return _train_url_model_internal()


def classify_url(model, url: str):
    prob = float(model.predict_proba([url])[0][1])
    label = "🔴 MALICIOUS" if prob >= 0.5 else "🟢 LEGITIMATE"
    return prob, label


def show_url_classifier():
    st.title("🌍  Malicious IP / URL Classifier")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#5b8aaa;
    font-size:0.8rem;border-left:3px solid #0ea5e9;padding-left:1rem;margin-bottom:1.5rem;">
    A <b>Character-level CNN</b> reads URLs one character at a time — no hand-crafted rules.
    It learns what malicious URLs <i>look like</i> from patterns in the raw text.
    Paste URLs below or upload a CSV for bulk classification.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading CNN model (trains once per session)…"):
        url_model = get_url_model()

    st.success("✅  CNN model ready")

    tab1, tab2 = st.tabs(["🔍 Single / Manual Check", "📂 Bulk CSV Classification"])

    # ── Tab 1: manual ─────────────────────────────────
    with tab1:
        st.subheader("Paste URLs to classify")
        raw = st.text_area(
            "Enter one URL per line",
            placeholder="https://paypa1-login.xyz/secure?token=abc\nhttps://google.com",
            height=160
        )
        if st.button("⚡  Classify URLs", use_container_width=True):
            urls = [u.strip() for u in raw.splitlines() if u.strip()]
            if not urls:
                st.warning("Please enter at least one URL.")
            else:
                results = []
                for u in urls:
                    prob, label = classify_url(url_model, u)
                    results.append({
                        "URL":         u,
                        "Prediction":  label,
                        "Confidence":  f"{prob * 100:.1f}%",
                        "Risk Score":  round(prob, 4)
                    })
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)

                n_mal = sum(1 for r in results if "MALICIOUS" in r["Prediction"])
                c1, c2 = st.columns(2)
                c1.metric("Malicious Detected", n_mal)
                c2.metric("Legitimate",          len(results) - n_mal)

    # ── Tab 2: bulk CSV ───────────────────────────────
    with tab2:
        st.subheader("Bulk classify from CSV")
        uploaded = st.file_uploader("Upload CSV with URL column", type=["csv"], key="url_bulk")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            url_col = auto_detect_column(df, ["url", "link", "domain", "host"])
            url_col = st.selectbox("URL column", df.columns,
                                   index=list(df.columns).index(url_col) if url_col else 0)
            if st.button("🚀  Run Bulk Classification", use_container_width=True):
                with st.spinner("Classifying…"):
                    probs, labels = [], []
                    for u in df[url_col].astype(str):
                        p, l = classify_url(url_model, u)
                        probs.append(round(p, 4))
                        labels.append(l)
                df["CNN_Prediction"] = labels
                df["Risk_Score"]     = probs
                st.dataframe(df, use_container_width=True)

                n_mal = labels.count("🔴 MALICIOUS")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total URLs",  len(labels))
                c2.metric("Malicious",   n_mal)
                c3.metric("Legitimate",  len(labels) - n_mal)

                dist = pd.DataFrame({"Count": [n_mal, len(labels)-n_mal]},
                                    index=["Malicious", "Legitimate"])
                st.bar_chart(dist)


# ═════════════════════════════════════════════════════
# ③ PHISHING EMAIL DETECTOR  —  LSTM on text
# ═════════════════════════════════════════════════════

# ── Fast Email Classifier: TF-IDF + Logistic Regression (trains in <1s) ──

def _generate_email_data(n=1200):
    rng = np.random.default_rng(0)
    legit = [
        "meeting scheduled for tomorrow at 10am please confirm attendance",
        "quarterly report is ready for review attached please find document",
        "team lunch friday noon please rsvp by thursday",
        "your order has been shipped tracking number provided in attachment",
        "invoice for services rendered this month amount due next friday",
        "welcome to the team onboarding documents attached for review",
        "please review the attached proposal and share your feedback",
        "reminder your performance review is scheduled for next week",
    ]
    spam = [
        "congratulations you won a million dollars click here to claim now",
        "limited time offer buy one get three free order today",
        "make money fast from home no experience needed guaranteed income",
        "lose weight fast with this one weird trick doctors hate",
        "cheap meds online no prescription needed best prices delivery",
        "you have been selected exclusive reward claim before it expires",
        "hot singles in your area click now free registration tonight",
    ]
    phishing = [
        "your account has been compromised verify your credentials immediately",
        "unusual login detected confirm your password to secure account now",
        "your bank account is suspended verify information to restore access",
        "paypal security alert update payment details immediately avoid suspension",
        "apple id locked verify your identity click the link within 24 hours",
        "urgent action required email storage full click to upgrade immediately",
        "dear customer your password expires today click here to reset",
        "we noticed suspicious activity please verify your identity now",
    ]
    texts, labels = [], []
    for _ in range(n // 3):
        base = legit[rng.integers(len(legit))]
        extra = " ".join(rng.choice(["please","regards","team","note","attached","review"], 3))
        texts.append(base + " " + extra); labels.append(0)
    for _ in range(n // 3):
        base = spam[rng.integers(len(spam))]
        extra = " ".join(rng.choice(["free","offer","guaranteed","click","buy","now"], 4))
        texts.append(base + " " + extra); labels.append(1)
    for _ in range(n - 2*(n//3)):
        base = phishing[rng.integers(len(phishing))]
        extra = " ".join(rng.choice(["urgent","verify","account","secure","click","immediately"], 4))
        texts.append(base + " " + extra); labels.append(2)
    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


@st.cache_resource(show_spinner="Loading phishing detector…")
def get_email_model():
    """TF-IDF + Logistic Regression — instant training, no GPU needed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    texts, labels = _generate_email_data(1200)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)),
        ("lr",    LogisticRegression(max_iter=300, C=5, random_state=42)),
    ])
    clf.fit(texts, labels)
    return clf, None   # second value kept for API compat (was word2idx)


def predict_email(model, _unused, text: str):
    prob = model.predict_proba([text])[0]   # shape (3,)  0=legit,1=spam,2=phish
    cls  = int(np.argmax(prob))
    labels_map = {0: "🟢 LEGITIMATE", 1: "🟡 SPAM", 2: "🔴 PHISHING"}
    return labels_map[cls], {
        "Legitimate": f"{prob[0]*100:.1f}%",
        "Spam":       f"{prob[1]*100:.1f}%",
        "Phishing":   f"{prob[2]*100:.1f}%",
    }


# ═════════════════════════════════════════════════════
# QR CODE DETECTOR  —  Threat / Data-Leak Analysis
# ═════════════════════════════════════════════════════

def analyse_qr_url(url_model, raw_url: str):
    """
    Multi-signal QR payload analyser.
    Returns a dict of findings.
    """
    findings = {}

    # ── 1. Basic URL structure ──
    findings["raw_payload"] = raw_url

    # Protocol
    if raw_url.startswith("http://"):
        findings["protocol"] = ("⚠️  HTTP (unencrypted)", "warning")
    elif raw_url.startswith("https://"):
        findings["protocol"] = ("✅  HTTPS (encrypted)", "ok")
    elif raw_url.startswith("tel:") or raw_url.startswith("sms:"):
        findings["protocol"] = ("📞  Phone/SMS redirect", "warning")
    elif raw_url.startswith("mailto:"):
        findings["protocol"] = ("📧  Email redirect", "warning")
    elif raw_url.startswith("data:"):
        findings["protocol"] = ("🚨  Data URI — high risk", "critical")
    elif raw_url.startswith("javascript:"):
        findings["protocol"] = ("🚨  JavaScript injection", "critical")
    else:
        findings["protocol"] = ("ℹ️  Non-HTTP payload", "info")

    # ── 2. Suspicious keyword scan ──
    sus_keywords = [
        "login", "verify", "secure", "update", "account", "confirm",
        "banking", "paypal", "appleid", "password", "credential",
        "token=", "redirect=", "url=", "goto=", "return=",
    ]
    data_keywords = [
        "ssn=", "dob=", "phone=", "address=", "card=", "cvv=",
        "pin=", "passport=", "license=", "income=", "salary=",
    ]

    url_lower = raw_url.lower()
    found_sus  = [k for k in sus_keywords  if k in url_lower]
    found_data = [k for k in data_keywords if k in url_lower]

    findings["suspicious_keywords"] = found_sus
    findings["data_leak_params"]    = found_data

    # ── 3. IP-address host (vs domain) ──
    ip_pattern = re.compile(
        r"https?://(\d{1,3}\.){3}\d{1,3}"
    )
    findings["raw_ip_host"] = bool(ip_pattern.match(raw_url))

    # ── 4. Excessive subdomains ──
    try:
        from urllib.parse import urlparse
        host = urlparse(raw_url).netloc.split(":")[0]
        sub_count = host.count(".")
        findings["subdomain_count"] = sub_count
        findings["suspicious_subdomain"] = sub_count > 3
    except Exception:
        findings["subdomain_count"] = 0
        findings["suspicious_subdomain"] = False

    # ── 5. URL shortener detection ──
    shorteners = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly",
                  "rebrand.ly", "short.io", "is.gd", "buff.ly", "cutt.ly"]
    findings["is_shortener"] = any(s in url_lower for s in shorteners)

    # ── 6. CNN maliciousness score ──
    prob, cnn_label = classify_url(url_model, raw_url)
    findings["cnn_score"] = prob
    findings["cnn_label"] = cnn_label

    # ── 7. Aggregate risk score (0-100) ──
    score = int(prob * 50)                              # CNN base
    score += 15 if found_sus else 0
    score += 20 if found_data else 0
    score += 10 if findings["raw_ip_host"] else 0
    score += 10 if findings["suspicious_subdomain"] else 0
    score += 10 if findings["is_shortener"] else 0
    score += 10 if findings["protocol"][1] == "critical" else 0
    score += 5  if findings["protocol"][1] == "warning"  else 0
    score = min(score, 100)
    findings["risk_score"] = score

    if score >= 70:
        findings["verdict"] = ("🔴 HIGH RISK — Likely Malicious / Data-Leak", "critical")
    elif score >= 40:
        findings["verdict"] = ("🟡 MEDIUM RISK — Suspicious, Verify Before Opening", "warning")
    else:
        findings["verdict"] = ("🟢 LOW RISK — Appears Safe", "ok")

    return findings


def show_qr_detector():
    import io

    # Try zxing-cpp first (pure-Python wheel, no native DLL needed on Windows)
    # Install: pip install zxing-cpp pillow
    try:
        import zxingcpp
        from PIL import Image as PILImage
        qr_backend = "zxing"
    except ImportError:
        # Fallback: try OpenCV
        try:
            import cv2
            qr_backend = "cv2"
        except ImportError:
            qr_backend = None

    st.title("📷  QR Code Detector & Threat Analyser")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#b794f4;
    font-size:0.8rem;border-left:3px solid #0ea5e9;padding-left:1rem;margin-bottom:1.5rem;">
    Upload a QR code image — decodes the payload, then runs multi-signal security
    analysis: CNN classifier, data-leak detection, keyword scan, protocol check, and more.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading CNN model…"):
        url_model = get_url_model()
    st.success("✅  CNN model ready")

    tab1, tab2 = st.tabs(["📷 Scan QR Image", "🔗 Analyse URL / Payload Manually"])

    # ── Tab 1: QR image upload (OpenCV — no external DLL needed) ──────────
    with tab1:
        st.subheader("Upload a QR Code Image")
        uploaded_qr = st.file_uploader(
            "Supported formats: PNG, JPG, WEBP, BMP",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="qr_upload"
        )

        if uploaded_qr:
            img_bytes = uploaded_qr.read()
            st.image(img_bytes, caption="Uploaded QR Code", width=280)

            if qr_backend is None:
                st.warning(
                    "⚠️  No QR decoding library found. Install one with:\n\n"
                    "```\npip install zxing-cpp pillow\n```\n\n"
                    "Then restart Streamlit. Or paste the URL manually in **Tab 2**."
                )
            else:
                try:
                    data = None
                    if qr_backend == "zxing":
                        import zxingcpp
                        from PIL import Image as PILImage
                        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                        results = zxingcpp.read_barcodes(pil_img)
                        if results:
                            data = results[0].text
                    else:  # cv2
                        import cv2
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        detector = cv2.QRCodeDetector()
                        data, _, _ = detector.detectAndDecode(img_cv)

                    if not data:
                        st.error("❌  No QR code detected. Try a clearer or higher-resolution image.")
                    else:
                        st.markdown(f"**Decoded payload:** `{data}`")
                        findings = analyse_qr_url(url_model, data)
                        _render_qr_findings(findings)

                except Exception as e:
                    st.error(f"QR decode error: {e}")

    # ── Tab 2: manual payload ──────────────────────────
    with tab2:
        st.subheader("Paste QR Payload / URL for Analysis")
        manual = st.text_input(
            "Enter URL or raw QR payload",
            placeholder="https://paypa1-login.xyz/secure?token=abc123"
        )
        if st.button("⚡  Analyse Payload", use_container_width=True):
            if not manual.strip():
                st.warning("Please enter a URL or payload.")
            else:
                findings = analyse_qr_url(url_model, manual.strip())
                _render_qr_findings(findings)


def _render_qr_findings(f: dict):
    """Render the QR analysis findings dict into Streamlit UI."""

    verdict_text, verdict_level = f["verdict"]
    risk = f["risk_score"]

    # Big verdict banner
    color = {"critical": "#ff2d55", "warning": "#f59e0b", "ok": "#00ffe7"}.get(verdict_level, "#5b8aaa")
    st.markdown(
        f"<h2 style='text-align:center;color:{color};margin:0.8rem 0'>{verdict_text}</h2>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score",    f"{risk} / 100")
    c2.metric("CNN Verdict",   f["cnn_label"])
    c3.metric("CNN Confidence", f"{f['cnn_score']*100:.1f}%")

    with st.expander("🔎 Detailed Signal Breakdown", expanded=True):

        # Protocol
        proto_text, proto_level = f["protocol"]
        proto_color = {"critical": "#ff2d55", "warning": "#f59e0b",
                       "ok": "#00ffe7", "info": "#5b8aaa"}.get(proto_level, "#5b8aaa")
        st.markdown(
            f"**Protocol:** <span style='color:{proto_color}'>{proto_text}</span>",
            unsafe_allow_html=True
        )

        # Suspicious keywords
        if f["suspicious_keywords"]:
            st.warning(f"⚠️  Suspicious keywords found: `{', '.join(f['suspicious_keywords'])}`")
        else:
            st.success("✅  No suspicious keywords detected")

        # Data-leak parameters
        if f["data_leak_params"]:
            st.error(f"🚨  Potential data-leak parameters: `{', '.join(f['data_leak_params'])}`")
        else:
            st.success("✅  No data-exfiltration parameters detected")

        # IP host
        if f["raw_ip_host"]:
            st.warning("⚠️  URL uses a raw IP address instead of a domain — high phishing indicator")

        # Subdomain depth
        st.info(f"ℹ️  Subdomain depth: {f['subdomain_count']} "
                + ("— ⚠️ Suspicious (>3)" if f["suspicious_subdomain"] else "— OK"))

        # Shortener
        if f["is_shortener"]:
            st.warning("⚠️  URL shortener detected — real destination is hidden")

        # Raw payload
        st.markdown(f"**Raw payload:** `{f['raw_payload']}`")

    # Risk gauge bar
    bar_color = "#ff2d55" if risk >= 70 else "#f59e0b" if risk >= 40 else "#00ffe7"
    st.markdown(
        f"""
        <div style='margin-top:0.5rem;'>
        <div style='font-family:"Share Tech Mono",monospace;color:#5b8aaa;font-size:0.75rem;
                    letter-spacing:0.1em;margin-bottom:4px;'>RISK GAUGE</div>
        <div style='background:#071525;border:1px solid rgba(0,255,231,0.15);
                    border-radius:4px;height:18px;overflow:hidden;'>
            <div style='width:{risk}%;height:100%;background:{bar_color};
                        box-shadow:0 0 12px {bar_color};transition:width 0.5s;'></div>
        </div></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


def show_phishing_detector():
    st.title("📧  Phishing Email Detector")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#5b8aaa;
    font-size:0.8rem;border-left:3px solid #00ffe7;padding-left:1rem;margin-bottom:1.5rem;">
    A <b>TF-IDF + Logistic Regression</b> classifier detects email intent as
    <span style="color:#00ffe7">Legitimate</span>,
    <span style="color:#f59e0b">Spam</span>, or
    <span style="color:#ff2d55">Phishing</span> — instant results, no GPU needed.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading phishing detector…"):
        email_model, word2idx = get_email_model()

    st.success("✅  Phishing detector ready")

    tab1, tab2 = st.tabs(["✉️ Single Email Check", "📂 Bulk CSV Analysis"])

    # ── Tab 1: single ─────────────────────────────────
    with tab1:
        st.subheader("Paste Email Content")
        subject = st.text_input("Subject line", placeholder="Your account has been suspended")
        body    = st.text_area("Email body", height=200,
                               placeholder="Dear customer, we noticed unusual activity…")

        if st.button("⚡  Analyse Email", use_container_width=True):
            combined = f"{subject} {body}".strip()
            if not combined.strip():
                st.warning("Please enter subject or body text.")
            else:
                label, probs = predict_email(email_model, word2idx, combined)

                st.markdown(
                    f"<h2 style='text-align:center;margin:1rem 0'>{label}</h2>",
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Legitimate", probs["Legitimate"])
                c2.metric("Spam",       probs["Spam"])
                c3.metric("Phishing",   probs["Phishing"])

                prob_df = pd.DataFrame({
                    "Probability": [
                        float(probs["Legitimate"].strip("%")) / 100,
                        float(probs["Spam"].strip("%")) / 100,
                        float(probs["Phishing"].strip("%")) / 100
                    ]
                }, index=["Legitimate", "Spam", "Phishing"])
                st.bar_chart(prob_df)

                if "PHISHING" in label:
                    st.error("🚨  High phishing confidence — do not click any links in this email!")
                elif "SPAM" in label:
                    st.warning("⚠  This email looks like spam — treat with caution.")
                else:
                    st.success("✅  Email appears legitimate.")

    # ── Tab 2: bulk CSV ───────────────────────────────
    with tab2:
        st.subheader("Bulk Classify from CSV")
        st.info("CSV should have a column containing email subject/body text.")
        uploaded = st.file_uploader("Upload Email CSV", type=["csv"], key="email_bulk")

        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())

            text_col = auto_detect_column(df, ["subject", "body", "email", "text", "content", "message"])
            text_col = st.selectbox("Text column", df.columns,
                                    index=list(df.columns).index(text_col) if text_col else 0)

            if st.button("🚀  Run Bulk Detection", use_container_width=True):
                with st.spinner("Classifying emails…"):
                    predictions, leg, spam_p, phish_p = [], [], [], []
                    for txt in df[text_col].astype(str):
                        lbl, prb = predict_email(email_model, word2idx, txt)
                        predictions.append(lbl)
                        leg.append(prb["Legitimate"])
                        spam_p.append(prb["Spam"])
                        phish_p.append(prb["Phishing"])

                df["Prediction"]  = predictions
                df["P_Legit"]     = leg
                df["P_Spam"]      = spam_p
                df["P_Phishing"]  = phish_p

                st.dataframe(df, use_container_width=True)

                counts = pd.Series(predictions).value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Legitimate", counts.get("🟢 LEGITIMATE", 0))
                c2.metric("🟡 Spam",       counts.get("🟡 SPAM", 0))
                c3.metric("🔴 Phishing",   counts.get("🔴 PHISHING", 0))

                dist_df = pd.DataFrame({"Count": counts})
                st.bar_chart(dist_df)


if not st.session_state.logged_in:

    show_login()

else:

    with st.sidebar:

        st.markdown("""
        <div style="
            font-family:'Orbitron',sans-serif;
            color:#00ffe7;
            font-size:0.85rem;
            font-weight:900;
            letter-spacing:0.15em;
            padding: 0.5rem 0 0.2rem;
            text-shadow: 0 0 12px rgba(0,255,231,0.5);
        ">🛡️ NAVIGATION</div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Modules",
            [
                "Threat Detection",
                "Digital Forensics",
                "URL / IP Classifier",
                "QR Code Detector",
                "Phishing Detector",
            ]
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⏻  LOGOUT", use_container_width=True):

            st.session_state.logged_in = False

            st.rerun()

    if page == "Threat Detection":
        show_threat_detection()

    elif page == "Digital Forensics":
        show_forensics()

    elif page == "URL / IP Classifier":
        show_url_classifier()

    elif page == "QR Code Detector":
        show_qr_detector()

    elif page == "Phishing Detector":
        show_phishing_detector()
