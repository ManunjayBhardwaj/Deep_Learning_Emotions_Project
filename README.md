# 🎵 Emotion Detection System (LSTM + ANN)

**Test Accuracy: 97.42%**

This project implements a **Hybrid Deep Learning Model** combining a BiLSTM for FFT-based sequential data and a Feed‑Forward ANN for statistical features. The system predicts emotional states using audio-derived features and provides a Streamlit-based interactive dashboard for visual exploration and prediction.

---

## 🚀 Features

### 🔥 Model Highlights
- **97.42% Test Accuracy**
- **BiLSTM (Bidirectional) for FFT sequence inputs**
- **ANN for statistical features**
- Hybrid fusion layer for final emotion classification

### 🖥 Streamlit App Features
- Upload CSV with **any number of rows** (batch predictions)
- Auto-detects FFT & statistical columns
- Generates:
  - Correlation heatmap  
  - FFT visualization  
  - Prediction distribution plot  
- Downloadable predictions CSV
- No manual inputs required

---

## 📁 Project Structure

```
emotion-detection/
│
├── app.py                 # Streamlit application
├── hybrid_lstm_ann.pth    # Trained model
├── stat_scaler.pkl        # Scaler for statistical features
├── label_encoder.pkl      # Label encoder for emotion classes
├── emotions.csv           # Sample dataset (for determining cols)
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### **LSTM Branch**
- Input: FFT sequence (750 × 1)
- 2‑layer BiLSTM (hidden = 64)
- Dropout 0.3
- Fully‑connected layer → 128‑dimensional embedding

### **ANN Branch**
- Input: statistical features
- Linear(⚙ dim → 128) → ReLU → Dropout  
- Linear(128 → 64) → ReLU

### **Fusion**
- Concatenated embedding: (128 + 64)
- Linear → ReLU → Dropout  
- Final Linear → Softmax

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📤 Deployment

### Streamlit Cloud
1. Push repository to GitHub  
2. Go to **https://share.streamlit.io**  
3. Select repository & app.py  

---

## 📜 License
MIT License.

---

### ⭐ If you like this project, please star the repository! ⭐
