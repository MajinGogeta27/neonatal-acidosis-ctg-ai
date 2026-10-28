import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from data_loader import CTUDataLoader
from feature_extractor import CTGFeatureExtractor
from models_engineered import FeatureModelSuite
from models_deep import DeepModelTrainer, CTG1DCNN, CTGCNNLSTM, CTGTransformer
from evaluator import ModelEvaluator
from explainability import FeatureExplainability, DeepTemporalSaliency


def save_checkpoint(results_dir, name, **arrays):
    """Save model predictions so completed models do not need to be retrained."""
    path = os.path.join(results_dir, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(results_dir, name):
    """Load a previously completed checkpoint."""
    path = os.path.join(results_dir, f"{name}.npz")

    if os.path.exists(path):
        print(f"  Loading existing checkpoint: {path}")
        return dict(np.load(path))

    return None


def main():

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("      PREDICTING NEONATAL ACIDOSIS FROM CTG WAVEFORMS (AI)")
    print("  Continuous Umbilical Artery pH Prediction & Threshold Classification")
    print("=" * 70)

    # ================================================================
    # Step 1: Load CTU-CHB Dataset
    # ================================================================

    print("\n--- Step 1: Loading Dataset ---")

    loader = CTUDataLoader()

    df_meta, fhr_signals, uc_signals = loader.load_dataset(
        window_minutes=30.0
    )

    y_ph = df_meta["pH"].values.astype(np.float32)
    y_710 = df_meta["pH_less_710"].values.astype(int)
    y_705 = df_meta["pH_less_705"].values.astype(int)

    # Compute class-imbalance weights for XGBoost (neg / pos)
    pos_710 = int(np.sum(y_710))
    neg_710 = int(len(y_710) - pos_710)
    pos_705 = int(np.sum(y_705))
    neg_705 = int(len(y_705) - pos_705)
    pos_weight_710 = neg_710 / max(pos_710, 1)
    pos_weight_705 = neg_705 / max(pos_705, 1)

    print("\nCohort Overview:")
    print(f"  Total Valid Records: {len(df_meta)}")
    print(
        f"  Continuous pH Mean ± SD: "
        f"{np.mean(y_ph):.3f} ± {np.std(y_ph):.3f} "
        f"(Range: {np.min(y_ph):.2f} - {np.max(y_ph):.2f})"
    )
    print(
        f"  pH < 7.10 Count (PRIMARY — Significant Acidemia): "
        f"{pos_710} ({pos_710 / len(y_710) * 100:.1f}%)  "
        f"[pos_weight = {pos_weight_710:.1f}]"
    )
    print(
        f"  pH < 7.05 Count (Secondary — Severe Acidemia): "
        f"{pos_705} ({pos_705 / len(y_705) * 100:.1f}%)  "
        f"[pos_weight = {pos_weight_705:.1f}]"
    )

    # ================================================================
    # Step 2: Feature Extraction
    # ================================================================

    print("\n--- Step 2: Feature Extraction ---")

    extractor = CTGFeatureExtractor()

    X_features, feature_names = extractor.extract_features_dataset(
        fhr_signals,
        uc_signals
    )

    for idx, fname in enumerate(feature_names):
        df_meta[fname] = X_features[:, idx]

    # ================================================================
    # Step 3: Engineered Feature Models
    # ================================================================

    print("\n--- Step 3: Engineered Feature Models ---")

    suite = FeatureModelSuite(
        n_splits=5,
        random_state=42
    )

    engineered_checkpoint = load_checkpoint(
        results_dir,
        "engineered_models"
    )

    if engineered_checkpoint is not None:

        oof_reg_preds = {
            "Ridge Regression":
                engineered_checkpoint["ridge"],
            "Random Forest Regressor":
                engineered_checkpoint["random_forest"],
            "HistGradientBoosting Regressor":
                engineered_checkpoint["hist_gradient_boosting"],
            "XGBoost Regressor":
                engineered_checkpoint["xgboost"],
        }

        fitted_reg_pipelines = {}

        print("  Engineered model predictions loaded from checkpoint.")

    else:

        oof_reg_preds, fitted_reg_pipelines = (
            suite.evaluate_cross_validation_regression(
                X_features,
                y_ph,
                y_710
            )
        )

        save_checkpoint(
            results_dir,
            "engineered_models",
            ridge=oof_reg_preds["Ridge Regression"],
            random_forest=oof_reg_preds["Random Forest Regressor"],
            hist_gradient_boosting=oof_reg_preds[
                "HistGradientBoosting Regressor"
            ],
            xgboost=oof_reg_preds["XGBoost Regressor"],
        )

    # ================================================================
    # Step 4: Deep Learning Models
    # ================================================================

    print("\n--- Step 4: Deep Learning Models ---")

    deep_trainer = DeepModelTrainer(
        n_splits=5,
        epochs=20,
        lr=1e-3
    )

    # ------------------------------------------------
    # 4A. 1D CNN
    # ------------------------------------------------

    cnn_checkpoint = load_checkpoint(
        results_dir,
        "cnn_predictions"
    )

    if cnn_checkpoint is not None:

        oof_cnn_ph = cnn_checkpoint["ph"]
        oof_cnn_710 = cnn_checkpoint["p710"]
        oof_cnn_705 = cnn_checkpoint["p705"]

        cnn_models = []

        print("  1D-CNN predictions loaded from checkpoint.")

    else:

        (
            oof_cnn_ph,
            oof_cnn_710,
            oof_cnn_705,
            cnn_models
        ) = deep_trainer.train_evaluate(
            CTG1DCNN,
            fhr_signals,
            uc_signals,
            y_ph,
            y_710,
            y_705,
            model_name="1D-CNN (PyTorch)"
        )

        save_checkpoint(
            results_dir,
            "cnn_predictions",
            ph=oof_cnn_ph,
            p710=oof_cnn_710,
            p705=oof_cnn_705
        )

    # ------------------------------------------------
    # 4B. CNN-LSTM
    # ------------------------------------------------

    lstm_checkpoint = load_checkpoint(
        results_dir,
        "cnn_lstm_predictions"
    )

    if lstm_checkpoint is not None:

        oof_lstm_ph = lstm_checkpoint["ph"]
        oof_lstm_710 = lstm_checkpoint["p710"]
        oof_lstm_705 = lstm_checkpoint["p705"]

        print("  CNN-LSTM predictions loaded from checkpoint.")

    else:

        (
            oof_lstm_ph,
            oof_lstm_710,
            oof_lstm_705,
            _
        ) = deep_trainer.train_evaluate(
            CTGCNNLSTM,
            fhr_signals,
            uc_signals,
            y_ph,
            y_710,
            y_705,
            model_name="CNN-LSTM (PyTorch)"
        )

        save_checkpoint(
            results_dir,
            "cnn_lstm_predictions",
            ph=oof_lstm_ph,
            p710=oof_lstm_710,
            p705=oof_lstm_705
        )

    # ------------------------------------------------
    # 4C. Transformer / Conformer
    # ------------------------------------------------

    transformer_checkpoint = load_checkpoint(
        results_dir,
        "transformer_predictions"
    )

    if transformer_checkpoint is not None:

        oof_trans_ph = transformer_checkpoint["ph"]
        oof_trans_710 = transformer_checkpoint["p710"]
        oof_trans_705 = transformer_checkpoint["p705"]

        print("  Transformer predictions loaded from checkpoint.")

    else:

        print("\n  Starting Transformer/Conformer training...")
        print("  This may take significantly longer on CPU.")

        (
            oof_trans_ph,
            oof_trans_710,
            oof_trans_705,
            _
        ) = deep_trainer.train_evaluate(
            CTGTransformer,
            fhr_signals,
            uc_signals,
            y_ph,
            y_710,
            y_705,
            model_name="1D-Transformer (Conformer)"
        )

        save_checkpoint(
            results_dir,
            "transformer_predictions",
            ph=oof_trans_ph,
            p710=oof_trans_710,
            p705=oof_trans_705
        )

    # Add deep-learning predictions to regression dictionary

    oof_reg_preds["PyTorch 1D-CNN"] = oof_cnn_ph
    oof_reg_preds["PyTorch CNN-LSTM"] = oof_lstm_ph
    oof_reg_preds["PyTorch 1D-Transformer"] = oof_trans_ph

    # ================================================================
    # Step 5: Continuous pH Regression
    # ================================================================

    evaluator = ModelEvaluator()

    reg_summary_list = []

    for name, pred in oof_reg_preds.items():

        res = evaluator.evaluate_regression(
            y_ph,
            pred,
            model_name=name
        )

        reg_summary_list.append(res)

    df_reg_summary = (
        pd.DataFrame(reg_summary_list)
        .sort_values(by="R2", ascending=False)
    )

    df_reg_summary.to_csv(
        os.path.join(
            results_dir,
            "continuous_ph_regression_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("           CONTINUOUS pH REGRESSION BENCHMARK")
    print("=" * 70)

    print(
        df_reg_summary[
            [
                "Model",
                "R2",
                "RMSE",
                "MAE",
                "Pearson_r",
                "Spearman_rho"
            ]
        ].to_string(index=False)
    )

    evaluator.plot_regression_scatter(
        y_ph,
        oof_reg_preds,
        save_path=os.path.join(
            results_dir,
            "regression_scatter.png"
        )
    )

    # ================================================================
    # Step 6: pH < 7.10 Classification  [PRIMARY THRESHOLD]
    # ================================================================

    print("\n" + "=" * 70)
    print("   PRIMARY THRESHOLD CLASSIFICATION BENCHMARK (pH < 7.10)")
    print("   N positive = %d / %d (%.1f%%) — most stable AUROC estimates" % (
        pos_710, len(y_710), pos_710 / len(y_710) * 100))
    print("=" * 70)

    oof_clf_710, _ = (
        suite.evaluate_cross_validation_classification(
            X_features,
            y_710,
            model_prefix="Direct",
            pos_weight=pos_weight_710,
        )
    )

    oof_clf_710["Direct PyTorch 1D-CNN"] = oof_cnn_710
    oof_clf_710["Direct PyTorch CNN-LSTM"] = oof_lstm_710

    # IMPORTANT:
    # Use the Transformer pH < 7.10 predictions here.
    oof_clf_710["Direct PyTorch 1D-Transformer"] = oof_trans_710

    # Thresholded regression predictions

    for reg_name, reg_preds in oof_reg_preds.items():

        prob_proxy = 1.0 / (
            1.0 + np.exp((reg_preds - 7.10) * 20.0)
        )

        oof_clf_710[
            f"Thresholded ({reg_name})"
        ] = prob_proxy

    clf_710_list = []

    for name, probs in oof_clf_710.items():

        m = evaluator.evaluate_classification(
            y_710,
            probs,
            threshold_val=0.5,
            model_name=name
        )

        clf_710_list.append(m)

    df_clf_710 = (
        pd.DataFrame(clf_710_list)
        .sort_values(by="AUROC", ascending=False)
    )

    df_clf_710.to_csv(
        os.path.join(
            results_dir,
            "threshold_classification_710_summary.csv"
        ),
        index=False
    )

    print(
        df_clf_710[
            [
                "Model",
                "AUROC",
                "AUPRC",
                "Sensitivity",
                "Specificity",
                "F1_Score"
            ]
        ].head(10).to_string(index=False)
    )

    # Bootstrap AUROC 95% CI for top-10 models
    print("\n  Bootstrap AUROC 95% CIs (pH < 7.10, PRIMARY):")
    for _, row in df_clf_710.head(10).iterrows():
        mname = row["Model"]
        probs = oof_clf_710.get(mname)
        if probs is not None:
            auroc, ci_lo, ci_hi = evaluator.bootstrap_auroc_ci(y_710, probs)
            print(f"    {mname:<45s}  AUROC {auroc:.3f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f})")

    # ROC curves: include Direct classifiers (HGB, LR, RF) + PyTorch models
    _roc_710_keys = [
        k for k in oof_clf_710
        if any(tag in k for tag in (
            "PyTorch", "Random Forest", "XGB", "LightGBM",
            "HistGradientBoosting", "Logistic"
        ))
    ]
    evaluator.plot_roc_curves(
        y_710,
        {k: oof_clf_710[k] for k in _roc_710_keys},
        title="ROC Curves for Significant Acidemia (pH < 7.10)",
        save_path=os.path.join(results_dir, "roc_710.png"),
    )

    # Confusion matrices — top 6 models by AUROC (pH < 7.10)
    evaluator.plot_confusion_matrices(
        y_710,
        {k: oof_clf_710[k] for k in _roc_710_keys},
        threshold_val=0.5,
        top_n=1,
        title="Confusion Matrices Significant Acidemia (pH < 7.10)",
        class_labels=["Normal pH\n(≥7.10)", "Acidemia\n(<7.10)"],
        save_path=os.path.join(results_dir, "confusion_matrices_710.png"),
    )

    # ================================================================
    # Step 7: pH < 7.05 Classification  [SECONDARY THRESHOLD]
    # ================================================================

    print("\n" + "=" * 70)
    print("   SECONDARY THRESHOLD CLASSIFICATION BENCHMARK (pH < 7.05)")
    print("   N positive = %d / %d (%.1f%%) — fewer events, wider CIs" % (
        pos_705, len(y_705), pos_705 / len(y_705) * 100))
    print("=" * 70)

    oof_clf_705, _ = (
        suite.evaluate_cross_validation_classification(
            X_features,
            y_705,
            model_prefix="Direct",
            pos_weight=pos_weight_705,
        )
    )

    oof_clf_705["Direct PyTorch 1D-CNN"] = oof_cnn_705
    oof_clf_705["Direct PyTorch CNN-LSTM"] = oof_lstm_705
    oof_clf_705["Direct PyTorch 1D-Transformer"] = oof_trans_705

    for reg_name, reg_preds in oof_reg_preds.items():

        prob_proxy = 1.0 / (
            1.0 + np.exp((reg_preds - 7.05) * 20.0)
        )

        oof_clf_705[
            f"Thresholded ({reg_name})"
        ] = prob_proxy

    clf_705_list = []

    for name, probs in oof_clf_705.items():

        m = evaluator.evaluate_classification(
            y_705,
            probs,
            threshold_val=0.5,
            model_name=name
        )

        clf_705_list.append(m)

    df_clf_705 = (
        pd.DataFrame(clf_705_list)
        .sort_values(by="AUROC", ascending=False)
    )

    df_clf_705.to_csv(
        os.path.join(
            results_dir,
            "threshold_classification_705_summary.csv"
        ),
        index=False
    )

    print(
        df_clf_705[
            [
                "Model",
                "AUROC",
                "AUPRC",
                "Sensitivity",
                "Specificity",
                "F1_Score"
            ]
        ].head(10).to_string(index=False)
    )

    # Bootstrap AUROC 95% CI for top-10 models
    print("\n  Bootstrap AUROC 95% CIs (pH < 7.05, SECONDARY — interpret with caution):")
    for _, row in df_clf_705.head(10).iterrows():
        mname = row["Model"]
        probs = oof_clf_705.get(mname)
        if probs is not None:
            auroc, ci_lo, ci_hi = evaluator.bootstrap_auroc_ci(y_705, probs)
            print(f"    {mname:<45s}  AUROC {auroc:.3f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f})")

    # ROC curves: include Direct classifiers (HGB, LR, RF) + PyTorch models
    _roc_705_keys = [
        k for k in oof_clf_705
        if any(tag in k for tag in (
            "PyTorch", "Random Forest", "XGB", "LightGBM",
            "HistGradientBoosting", "Logistic"
        ))
    ]
    evaluator.plot_roc_curves(
        y_705,
        {k: oof_clf_705[k] for k in _roc_705_keys},
        title="ROC Curves for Severe Acidemia (pH < 7.05)",
        save_path=os.path.join(results_dir, "roc_705.png"),
    )

    # Confusion matrices — top 6 models by AUROC (pH < 7.05)
    evaluator.plot_confusion_matrices(
        y_705,
        {k: oof_clf_705[k] for k in _roc_705_keys},
        threshold_val=0.5,
        top_n=6,
        title="Confusion Matrices for Severe Acidemia (pH < 7.05)",
        class_labels=["Normal pH\n(≥7.05)", "Severe Acidemia\n(<7.05)"],
        save_path=os.path.join(results_dir, "confusion_matrices_705.png"),
    )

    # ================================================================
    # Step 8: Categorical CTG Baseline
    # ================================================================

    cat_710, cat_705 = evaluator.compute_categorical_ctg_baseline(
        df_meta,
        y_710,
        y_705
    )

    print("\n" + "=" * 70)
    print("      CATEGORICAL CTG BASELINE (FIGO Status Quo Read)")
    print("=" * 70)

    print(
        f"  pH < 7.10 — AUROC: {cat_710['AUROC']:.3f}, "
        f"Sensitivity: {cat_710['Sensitivity']:.3f}, "
        f"Specificity: {cat_710['Specificity']:.3f}"
    )

    print(
        f"  pH < 7.05 — AUROC: {cat_705['AUROC']:.3f}, "
        f"Sensitivity: {cat_705['Sensitivity']:.3f}, "
        f"Specificity: {cat_705['Specificity']:.3f}"
    )

    # ================================================================
    # Step 9: Explainability
    # ================================================================

    print("\n--- Generating Explainability Artifacts ---")

    # Only run RF explainability if fitted models are available.

    if "Random Forest Regressor" in fitted_reg_pipelines:

        best_rf_pipeline = fitted_reg_pipelines[
            "Random Forest Regressor"
        ][0]

        rf_model = best_rf_pipeline.named_steps["model"]

        explain_feat = FeatureExplainability(
            feature_names
        )

        explain_feat.compute_shap_values(
            rf_model,
            X_features[:100],
            save_plot_path=os.path.join(
                results_dir,
                "feature_importance_shap.png"
            )
        )

    else:

        print(
            "  WARNING: RF fitted pipeline unavailable because "
            "engineered predictions were loaded from checkpoint."
        )

    # Deep saliency
    if len(cnn_models) > 0:
        min_ph_idx = int(np.argmin(y_ph))
        fold_i = [i for i, v in enumerate(deep_trainer.fold_val_indices) if min_ph_idx in v][0]
        saliency_engine = DeepTemporalSaliency(cnn_models[fold_i])
        saliency_engine.plot_temporal_saliency(
            fhr_signals[min_ph_idx], uc_signals[min_ph_idx],
            rec_id=df_meta.iloc[min_ph_idx]["rec_id"], actual_ph=float(y_ph[min_ph_idx]),
            save_path=os.path.join(results_dir, "deep_temporal_saliency.png"))

    # ================================================================
    # Step 10: Save Prediction Dataset
    # ================================================================

    print("\n--- Saving Prediction Dataset ---")

    df_preds = pd.DataFrame({

        "rec_id":
            df_meta["rec_id"],

        "actual_pH":
            y_ph,

        "actual_710":
            y_710,

        "actual_705":
            y_705,

        "pred_pH_RF":
            oof_reg_preds["Random Forest Regressor"],

        "pred_pH_1DCNN":
            oof_reg_preds["PyTorch 1D-CNN"],

        "pred_pH_CNNLSTM":
            oof_reg_preds["PyTorch CNN-LSTM"],

        "pred_pH_Transformer":
            oof_reg_preds["PyTorch 1D-Transformer"],

        "prob_710_1DCNN":
            oof_cnn_710,

        "prob_705_1DCNN":
            oof_cnn_705

    })

    df_preds.to_csv(
        os.path.join(
            results_dir,
            "predictions_dataset.csv"
        ),
        index=False
    )

    np.savez_compressed(
        os.path.join(
            results_dir,
            "signals_dataset.npz"
        ),
        fhr=fhr_signals,
        uc=uc_signals
    )

    # ================================================================
    # Finished
    # ================================================================

    print("\n" + "=" * 70)
    print("      ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(
        f"      Results saved to: "
        f"{os.path.abspath(results_dir)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()