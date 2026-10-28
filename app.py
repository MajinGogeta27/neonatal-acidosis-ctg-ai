from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="FETAL AI: Predicting Neonatal Acidosis from CTG",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 800; color: #5B2A86; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
    .card-stat { background-color: #F4EDF8; border-radius: 8px; padding: 1rem; border-left: 4px solid #7E4C90; }
    .card-stat-title { font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; }
    .card-stat-value { font-size: 1.8rem; font-weight: 700; color: #2B1040; }
    .card-stat-desc { font-size: 0.8rem; color: #777; }
</style>
""", unsafe_allow_html=True)

# Path relative to this file, so it works locally and on Streamlit Cloud
RESULTS_DIR = Path(__file__).parent / "results"


@st.cache_data
def load_results_data():
    reg_summary = pd.read_csv(RESULTS_DIR / "continuous_ph_regression_summary.csv", skipinitialspace=True)
    clf_710 = pd.read_csv(RESULTS_DIR / "threshold_classification_710_summary.csv", skipinitialspace=True)
    clf_705 = pd.read_csv(RESULTS_DIR / "threshold_classification_705_summary.csv", skipinitialspace=True)
    preds_df = pd.read_csv(RESULTS_DIR / "predictions_dataset.csv", skipinitialspace=True)

    # Defensively strip any leading/trailing whitespace from column names
    for df in [reg_summary, clf_710, clf_705, preds_df]:
        df.columns = df.columns.str.strip()

    signals = None
    sig_path = RESULTS_DIR / "signals_dataset.npz"
    if sig_path.exists():
        data = np.load(sig_path)
        signals = {"fhr": data["fhr"], "uc": data["uc"]}
    return reg_summary, clf_710, clf_705, preds_df, signals


def show_image(filename, caption):
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} not found in the results folder.")


def stat_card(title, value, desc):
    st.markdown(
        f"<div class='card-stat'><div class='card-stat-title'>{title}</div>"
        f"<div class='card-stat-value'>{value}</div>"
        f"<div class='card-stat-desc'>{desc}</div></div>",
        unsafe_allow_html=True,
    )


def main():
    st.markdown("<div class='main-header'>Predicting Neonatal Acidosis from CTG (AI)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Continuous umbilical artery pH prediction and threshold acidemia "
        "classification (pH &lt; 7.10 and pH &lt; 7.05). Research prototype, not for clinical use.</div>",
        unsafe_allow_html=True,
    )

    if not (RESULTS_DIR / "predictions_dataset.csv").exists():
        st.warning("Results data not found. Run the pipeline first (`python main.py`) and make sure the "
                   "`results` folder is in the same folder as this app.")
        return

    reg_summary, clf_710, clf_705, preds_df, signals = load_results_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Executive Summary",
        "Interactive CTG Inspector",
        "Continuous pH Regression Benchmark",
        "Threshold Classification Benchmark",
        "Explainability (SHAP and Saliency)",
    ])

    # ------------------------------------------------------------------
    if page == "Executive Summary":
        st.header("Executive Summary")
        st.markdown("""
        ### Research Question
        Can an AI model trained on continuous intrapartum CTG waveforms (FHR and uterine contractions)
        predict **umbilical artery pH at delivery**, both as a continuous lab value and as threshold flags
        (`pH < 7.10` and `pH < 7.05`), and how does it compare with a categorical FIGO-style CTG read?

        ### Key Features
        1. **Objective ground truth**: predicts lab-confirmed cord blood gas pH instead of clinician labels.
        2. **Multi-task framing**: continuous pH regression, direct binary classifiers, and thresholded regressors.
        3. **Dual AI pipeline**: engineered CTG features (baseline FHR, variability, decelerations,
           contractions, spectral measures) versus PyTorch deep models (1D-CNN, CNN-LSTM, 1D Transformer).
        4. **Explainability**: SHAP feature attributions and gradient saliency maps showing which parts of
           the tracing most influenced the deep model's prediction.
        """)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card("Total CTG Records", f"{len(preds_df)}", "CTU-CHB Intrapartum DB")
        with c2:
            stat_card("Best Continuous pH R²", f"{reg_summary['R2'].max():.3f}",
                      reg_summary.sort_values("R2", ascending=False).iloc[0]["Model"])
        with c3:
            stat_card("pH &lt; 7.10 AUROC", f"{clf_710['AUROC'].max():.3f}", "Significant acidemia")
        with c4:
            stat_card("pH &lt; 7.05 AUROC", f"{clf_705['AUROC'].max():.3f}", "Severe acidemia")

        st.subheader("Cohort Outcome Distribution")
        ph = preds_df["actual_pH"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(ph, kde=True, ax=ax[0], color="#7E4C90", bins=30)
        ax[0].axvline(7.10, color="orange", linestyle="--", label="pH = 7.10")
        ax[0].axvline(7.05, color="red", linestyle="--", label="pH = 7.05")
        ax[0].set_title("Cord Blood pH Distribution", fontsize=11, fontweight="bold")
        ax[0].set_xlabel("Umbilical Artery pH")
        ax[0].legend()

        counts = [
            (ph >= 7.15).sum(),
            ((ph < 7.15) & (ph >= 7.10)).sum(),
            ((ph < 7.10) & (ph >= 7.05)).sum(),
            (ph < 7.05).sum(),
        ]
        labels = ["Normal (≥ 7.15)", "Borderline (7.10 to 7.15)",
                  "Significant (7.05 to 7.10)", "Severe (< 7.05)"]
        ax[1].pie(counts, labels=labels, autopct="%1.1f%%",
                  colors=["#DCC4E4", "#B98ACB", "#7E4C90", "#4A1F6B"], startangle=140,
                  textprops={"fontsize": 9})
        ax[1].set_title("Acidemia Risk Categories", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    elif page == "Interactive CTG Inspector":
        st.header("Interactive CTG Waveform and Prediction Inspector")
        st.write("Select a record to view its 4 Hz FHR and uterine activity signal alongside the model's predictions.")

        rec_ids = preds_df["rec_id"].astype(str).tolist()
        selected_rec = st.selectbox("Select record ID:", rec_ids)
        rec_idx = preds_df[preds_df["rec_id"].astype(str) == selected_rec].index[0]
        rec = preds_df.iloc[rec_idx]

        p710, p705 = rec["prob_710_1DCNN"], rec["prob_705_1DCNN"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual cord blood pH", f"{rec['actual_pH']:.2f}")
        c2.metric("Predicted pH (1D-CNN)", f"{rec['pred_pH_1DCNN']:.2f}")
        c3.metric("pH < 7.10 risk", f"{p710 * 100:.1f}%",
                  delta="HIGH RISK" if p710 > 0.5 else "-LOW RISK", delta_color="inverse")
        c4.metric("pH < 7.05 risk", f"{p705 * 100:.1f}%",
                  delta="HIGH RISK" if p705 > 0.5 else "-LOW RISK", delta_color="inverse")

        if signals is not None:
            fhr, uc = signals["fhr"][rec_idx], signals["uc"][rec_idx]
            t = np.arange(len(fhr)) / (4.0 * 60.0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.plot(t, fhr, color="#7E4C90", lw=1.2, label="FHR (bpm)")
            ax1.axhline(110, color="gray", linestyle=":", alpha=0.6, label="Normal range (110 to 160 bpm)")
            ax1.axhline(160, color="gray", linestyle=":", alpha=0.6)
            ax1.set_ylabel("FHR (bpm)")
            ax1.set_title(f"Record {selected_rec}: 30 Minute CTG Tracing", fontsize=12, fontweight="bold")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(t, uc, color="#B98ACB", lw=1.2, label="Uterine activity (UC)")
            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("UC (relative units)")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Signal file not found, so the waveform cannot be shown.")

    # ------------------------------------------------------------------
    elif page == "Continuous pH Regression Benchmark":
        st.header("Continuous pH Regression Performance")
        st.write("AI models trained to predict exact umbilical artery pH at delivery.")
        st.dataframe(
            reg_summary.style
                .highlight_max(subset=["R2", "Pearson_r", "Spearman_rho"], color="#E6D6EE")
                .highlight_min(subset=["RMSE", "MAE"], color="#E6D6EE"),
            use_container_width=True,
        )
        st.subheader("Predicted vs Actual pH")
        show_image("regression_scatter.png", "Out-of-fold predicted pH versus lab-measured pH.")

    # ------------------------------------------------------------------
    elif page == "Threshold Classification Benchmark":
        st.header("Threshold Acidemia Classification Performance")
        st.write("Direct binary classifiers versus thresholded regressors.")
        tab1, tab2 = st.tabs(["Significant (pH < 7.10)", "Severe (pH < 7.05)"])
        cols = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "F1_Score"]
        with tab1:
            st.dataframe(clf_710.style.highlight_max(subset=cols, color="#E6D6EE"), use_container_width=True)
            show_image("roc_710.png", "ROC curves for pH < 7.10.")
        with tab2:
            st.dataframe(clf_705.style.highlight_max(subset=cols, color="#E6D6EE"), use_container_width=True)
            show_image("roc_705.png", "ROC curves for pH < 7.05.")

    # ------------------------------------------------------------------
    elif page == "Explainability (SHAP and Saliency)":
        st.header("Explainability")
        st.write("Tracing predictions back to engineered features and to regions of the raw waveform.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Engineered Feature Importance (SHAP)")
            show_image("feature_importance_shap.png", "Top engineered CTG features driving pH prediction.")
        with c2:
            st.subheader("Deep Learning Temporal Saliency")
            show_image("deep_temporal_saliency.png",
                       "Regions of the tracing that most influenced the 1D-CNN's predicted pH.")


if __name__ == "__main__":
    main()