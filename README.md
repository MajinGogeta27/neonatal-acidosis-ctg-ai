# Predicting Neonatal Acidosis from CTG (AI)
### Continuous Umbilical Artery pH Prediction & Threshold Acidemia Classification from Intrapartum Cardiotocography

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Clinical Motivation & Research Question

Intrapartum cardiotocography (CTG)—continuous monitoring of fetal heart rate (FHR) and uterine contractions (UC)—is the standard of care for monitoring fetal wellbeing during labor. However, standard clinical practice relies on categorizing tracings into subjective classes (*normal*, *suspicious*, *pathological* via FIGO/ACOG/NICE guidelines), which suffer from notoriously high inter- and intra-observer variability.

**This project reframes CTG interpretation around an objective, lab-confirmed biological ground truth:**
> **Research Question:** Can machine learning and deep learning models trained on 4 Hz CTG waveforms predict **umbilical artery pH at delivery directly**—both as a continuous value and at guideline-relevant acidosis thresholds ($\text{pH} < 7.10$ and $\text{pH} < 7.05$)—and how much clinical diagnostic power is lost by relying on subjective categorical classifications?

### Clinical Significance of Outcomes:
1. **Continuous Umbilical Artery pH (Primary Target):** Lab-measured cord blood gas at delivery. Preserves full physiological granularity across the viability spectrum.
2. **$\text{pH} < 7.10$ (Significant Acidemia):** Widely recognized guideline threshold requiring close neonatal monitoring.
3. **$\text{pH} < 7.05$ (Severe Acidemia):** High-severity threshold strongly correlated with hypoxic-ischemic encephalopathy (HIE), cerebral palsy, and neonatal intensive care admission.

---

## 📊 Key Benchmark Findings

All models were evaluated under **5-Fold Stratified Cross-Validation** on the full 552-record CTU-CHB Intrapartum Database.

### 1. Objective AI vs. Subjective Tracing Status Quo

| Target Cutoff | Model / Read Source | AUROC | AUPRC | Sensitivity | Specificity | Sens @ 80% Spec | Sens @ 90% Spec |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Significant Acidemia ($\text{pH} < 7.10$)** | **Direct Random Forest Classifier** | **0.767** | **0.324** | 0.232 | 0.962 | **0.607** | **0.464** |
| | **Thresholded RF Regressor** | **0.758** | **0.347** | 0.071 | 0.992 | **0.625** | **0.518** |
| | Direct XGBoost Classifier | 0.741 | 0.278 | 0.321 | 0.935 | 0.536 | 0.393 |
| | Direct Logistic Regression | 0.719 | 0.311 | 0.589 | 0.758 | 0.518 | 0.375 |
| | Direct PyTorch 1D-CNN | 0.615 | 0.134 | 0.000 | 1.000 | 0.232 | 0.143 |
| | *Subjective Categorical Read (FIGO Baseline)* | *0.560* | *0.139* | *0.000* | *1.000* | *0.339* | *0.161* |
| **Severe Acidemia ($\text{pH} < 7.05$)** | **Direct Random Forest Classifier** | **0.799** | **0.302** | 0.225 | 0.977 | **0.575** | **0.450** |
| | **Direct XGBoost Classifier** | **0.798** | **0.234** | 0.200 | 0.965 | **0.625** | **0.325** |
| | **Direct HistGradientBoosting** | **0.792** | **0.342** | 0.450 | 0.912 | **0.625** | **0.475** |
| | Thresholded RF Regressor | 0.739 | 0.307 | 0.000 | 1.000 | 0.625 | 0.550 |
| | Direct PyTorch 1D-CNN | 0.657 | 0.101 | 0.000 | 1.000 | 0.225 | 0.125 |
| | *Subjective Categorical Read (FIGO Baseline)* | *0.584* | *0.112* | *0.000* | *1.000* | *0.375* | *0.175* |

> **Takeaway:** Relying on subjective clinical reads costs **over 0.21 AUROC points** in discriminating neonatal acidemia. Objective AI models capture substantial physiological warning signs embedded in the raw CTG signal that categorical visual inspection overlooks.

### 2. Continuous pH Regression

| Model | $R^2$ | RMSE | MAE | Pearson $r$ ($p$-value) | Spearman $\rho$ ($p$-value) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Regressor** | **0.145** | **0.097** | **0.075** | **0.386** ($5.3 \times 10^{-21}$) | **0.348** ($3.9 \times 10^{-17}$) |
| **Ridge Regression** | 0.130 | 0.098 | 0.075 | 0.372 ($1.3 \times 10^{-19}$) | 0.351 ($1.9 \times 10^{-17}$) |
| **XGBoost Regressor** | 0.022 | 0.104 | 0.080 | 0.285 ($8.9 \times 10^{-12}$) | 0.264 ($2.9 \times 10^{-10}$) |
| **HistGradientBoosting** | -0.037 | 0.107 | 0.082 | 0.271 ($8.8 \times 10^{-11}$) | 0.234 ($2.6 \times 10^{-8}$) |

---

## 🧠 Physiological Explainability

### 1. SHAP Feature Attribution
- **Deceleration Area (AUC) & Maximum Deceleration Depth:** Dominant drivers reducing predicted umbilical artery pH.
- **Short-Term Variability (STV) & LTV:** Baseline variability collapse strongly drives acidosis classification.
- **Contraction Burden:** High uterine contraction frequency (tachysystole) with insufficient recovery windows accelerates fetal de-oxygenation.

### 2. Deep Temporal Saliency
Gradient-weighted backpropagation through PyTorch 1D-CNNs reveals that the neural network attributes highest importance to **repetitive late deceleration nadirs** lagging peak uterine contractions.

---

## 🖥️ Interactive Streamlit Dashboard

An interactive dashboard is included for exploring results and inspecting patient tracings:

```bash
# Launch interactive application
streamlit run app.py
```

### Dashboard Capabilities:
1. **Executive Summary:** Cohort outcome distributions, key performance metrics, and clinical thresholds.
2. **Interactive CTG Inspector:** Synchronized 30-minute dual-channel waveform viewer (FHR + UC) with live model predictions and alert badges.
3. **Continuous pH Regression Benchmark:** Detailed error metrics ($R^2$, RMSE, MAE, Pearson $r$) and predicted-vs-actual scatter plots.
4. **Threshold Classification Benchmark:** AUROC/AUPRC ranking, ROC curves, and confusion matrices for $\text{pH} < 7.10$ and $\text{pH} < 7.05$.
5. **Clinical Explainability:** Global SHAP feature importances and raw waveform gradient saliency overlays.

---

## 🚀 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/MajinGogeta27/neonatal-acidosis-ctg-ai.git
cd neonatal-acidosis-ctg-ai

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full cross-validation pipeline
python run_pipeline.py

# Launch Streamlit dashboard
streamlit run app.py
```

---

## 📁 Repository Structure

```text
├── data_loader.py            # CTU-CHB signal loader & header parser
├── feature_extractor.py      # FIGO-based clinical CTG feature engineering (27 features)
├── models_engineered.py      # Classical ML suite (Ridge, RF, XGBoost, LightGBM, HistGB)
├── models_deep.py            # PyTorch architectures (1D-CNN, CNN-LSTM, 1D-Transformer)
├── evaluator.py              # Cross-validation metrics, bootstrap CIs, ROC/scatter plots
├── explainability.py         # SHAP attributions & deep temporal gradient saliency
├── run_pipeline.py           # End-to-end master training & evaluation pipeline
├── app.py                    # Streamlit interactive clinical dashboard
├── recordings/               # Recorded video walkthrough of the application
├── results/                  # Precomputed predictions, model weights, and benchmark plots
└── requirements.txt          # Python package dependencies
```

---

## 📜 Dataset Reference
- **Database:** CTU-CHB Intrapartum Cardiotocography Database (PhysioNet).
- **Cohort:** 552 intrapartum recordings acquired at the University Hospital in Brno, Czech Republic, sampled at 4 Hz, paired with cord blood gas outcomes.
