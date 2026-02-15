# 🧠 Transparent Credit Card Fraud Detection

**An Interpretable Deep Learning Framework using LTACNN and Kolmogorov–Arnold Networks**

## 📌 Overview

Credit card fraud detection demands **high accuracy** while maintaining **model transparency**, especially in financial and regulatory environments. While deep learning models excel at capturing complex transaction patterns, they often act as *black boxes*, limiting trust and explainability.

This project proposes a **novel transparent fraud detection framework** that combines a high-performance **Linear Time Attention CNN-LSTM (LTACNN)** model with an inherently interpretable **Kolmogorov–Arnold Network (KAN)**. The result is a system that delivers **robust fraud detection performance** while offering **clear mathematical explanations** for its decisions.

## 🎯 Objectives

* Develop a hybrid **LTACNN-KAN framework** for credit card fraud detection
* Achieve high classification performance on imbalanced transaction data
* Introduce **model transparency and interpretability** without sacrificing accuracy
* Compare deep learning and KAN-based approaches using standard evaluation metrics
* Validate explanation consistency using **SHAP** and **LIME** techniques

## 🧩 Methodology

### 🔹 Data Preprocessing

* **Imbalance Handling:** Adaptive Synthetic Sampling (ADASYN)
* **Feature Scaling:** RobustScaler to reduce outlier influence
* **Data Transformation:** Converted into PyTorch tensors for deep learning compatibility

### 🔹 Model Architecture

* **LTACNN Model**

  * 1D CNN layers for local pattern extraction
  * LSTM layers for temporal sequence learning
  * Linear Attention mechanism for efficient feature weighting

* **KAN Model**

  * Implemented using the `pykan` library
  * Learns interpretable mathematical representations
  * Trained to approximate LTACNN behavior

### 🔹 Training

* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
* Framework: PyTorch
* Hardware: GPU-accelerated training

## 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Statistical comparison is performed using the **Wilcoxon signed-rank test** to assess significance between models.

## 📈 Results

| Model                   | Accuracy | Precision | Recall |
| ----------------------- | -------- | --------- | ------ |
| **LTACNN + Attention**  | 0.9434   | 0.9880    | 0.9782 |
| **KAN (Approximation)** | 0.9234   | 0.9790    | 0.9701 |

✔ KAN demonstrates **competitive performance** while providing **clear, interpretable outputs**
✔ Faster convergence and stable learning observed in KAN
✔ Key fraud-related features consistently identified across models

## 🔍 Explainability & Transparency

* **SHAP & LIME** used to analyze feature contributions
* KAN produces **explicit mathematical functions** instead of opaque activations
* Ensures interpretability suitable for **financial auditing and compliance**

## 🏗️ Project Structure

```bash
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── ltacnn.py
│   └── kan_model.py
├── training/
│   ├── train_ltacnn.py
│   └── train_kan.py
├── evaluation/
│   ├── metrics.py
│   └── explainability.py
├── results/
│   ├── plots/
│   └── logs/
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **NumPy / Pandas**
* **scikit-learn**
* **pykan**
* **SHAP & LIME**
* **Matplotlib / Seaborn**

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train LTACNN model
python training/train_ltacnn.py

# Train KAN model
python training/train_kan.py

# Evaluate models
python evaluation/metrics.py
```

## 🧪 Dataset

* Public credit card transaction dataset
* Highly imbalanced fraud vs non-fraud classes
* Preprocessed using ADASYN and robust scaling

## 📚 References

* Kolmogorov–Arnold Networks (KANs)
* Explainable AI (XAI) methods: SHAP, LIME
* Deep learning approaches for financial fraud detection

## 👨‍💻 Authors

* **Saksham Sharma**
* **Sushant Gargi**

School of Computer Science and Engineering
Vellore Institute of Technology, Chennai

## ⭐ Why This Project Stands Out

✔ Combines **accuracy + interpretability**
✔ Suitable for **real-world financial systems**
✔ Strong research orientation
✔ Recruiter-friendly & publication-ready
