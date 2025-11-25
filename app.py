import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title {
            font-size:42px; 
            font-weight:900; 
            text-align:center;
            color:#4A4A4A;
        }
        .card {
            background-color:#ffffff;
            padding:20px;
            border-radius:18px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.10);
        }
        .label-result {
            font-size:35px;
            font-weight:800;
            color:#1A73E8;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SCALER + LABEL ENCODER
# ============================================================
stat_scaler = joblib.load("stat_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class HybridModel(nn.Module):
    def __init__(self, seq_len, stat_dim, num_classes):
        super(HybridModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.lstm_fc = nn.Linear(64 * 2, 128)

        self.ann = nn.Sequential(
            nn.Linear(stat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, seq_input, stat_input):
        lstm_out, _ = self.lstm(seq_input)
        lstm_last = lstm_out[:, -1, :]
        lstm_feat = self.lstm_fc(lstm_last)
        ann_feat = self.ann(stat_input)
        combined = torch.cat([lstm_feat, ann_feat], dim=1)
        return self.classifier(combined)

# ============================================================
# LOAD SAMPLE TO IDENTIFY COLUMNS
# ============================================================
df_sample = pd.read_csv("emotions.csv")
seq_cols = [c for c in df_sample.columns if "fft_" in c]
stat_cols = [c for c in df_sample.columns if c not in seq_cols + ["label"]]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridModel(
    seq_len=len(seq_cols),
    stat_dim=len(stat_cols),
    num_classes=len(label_encoder.classes_)
).to(DEVICE)

model.load_state_dict(torch.load("hybrid_lstm_ann.pth", map_location=DEVICE))
model.eval()

# ============================================================
# MAIN TITLE
# ============================================================
st.markdown("<h1 class='main-title'>🎵 Emotion Detection System (LSTM + ANN)</h1>", unsafe_allow_html=True)
st.write("Upload a dataset and explore predictions + automatic visual insights.")

st.write("---")

# ============================================================
# UPLOAD CSV
# ============================================================
uploaded = st.file_uploader("📂 Upload CSV with FFT + Statistical Features", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Uploaded successfully with **{df.shape[0]} rows** and **{df.shape[1]} columns**")

    # Show preview
    st.write("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    # ======================================================
    # VISUALIZATION 1 — STAT FEATURE HEATMAP
    # ======================================================
    st.write("### 📊 Statistical Feature Correlation Map")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[stat_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Could not compute correlation heatmap.")

    # ======================================================
    # VISUALIZATION 2 — RANDOM FFT SIGNAL
    # ======================================================
    if len(df) > 0:
        st.write("### 🎧 Sample FFT Plot")
        sample_fft = df.iloc[0][seq_cols].values

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_fft)
        ax.set_title("FFT Amplitude Spectrum (Row 1)")
        st.pyplot(fig)

    # ======================================================
    # PREPROCESSING
    # ======================================================
    seq_input = df[seq_cols].to_numpy().astype(np.float32).reshape(len(df), len(seq_cols), 1)
    stat_input = df[stat_cols].to_numpy().astype(np.float32)
    stat_input = stat_scaler.transform(stat_input)

    seq_tensor = torch.tensor(seq_input).to(DEVICE)
    stat_tensor = torch.tensor(stat_input).to(DEVICE)

    # ======================================================
    # PREDICTION
    # ======================================================
    with torch.no_grad():
        outputs = model(seq_tensor, stat_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        pred_labels = label_encoder.inverse_transform(preds)

    df["predicted_label"] = pred_labels

    st.write("### 🎯 Predictions")
    st.dataframe(df[["predicted_label"]])

    # ======================================================
    # VISUALIZATION 3 — PREDICTION DISTRIBUTION
    # ======================================================
    st.write("### 📈 Prediction Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    df["predicted_label"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # ======================================================
    # DOWNLOAD RESULTS
    # ======================================================
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results CSV", csv_out, "predictions.csv", "text/csv")

