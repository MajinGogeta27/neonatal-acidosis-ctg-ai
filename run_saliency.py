import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from data_loader import CTUDataLoader
from models_deep import CTG1DCNN, DeepModelTrainer, PH_MIN, PH_MAX
from explainability import DeepTemporalSaliency


def diagnostics(y_ph, oof_ph, df_meta, results_dir, n_show=10):
    err = oof_ph - y_ph
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    base = np.mean(np.abs(y_ph - np.mean(y_ph)))
    r = np.corrcoef(y_ph, oof_ph)[0, 1]

    print("\n" + "=" * 70)
    print("  OUT-OF-FOLD pH DIAGNOSTICS")
    print("=" * 70)
    print(f"  MAE {mae:.3f} | RMSE {rmse:.3f} | bias {bias:+.3f} | Pearson r {r:.3f}")
    print(f"  Baseline MAE (always predict mean pH): {base:.3f}")
    print("  -> model beats baseline" if mae < base else
          "  -> WARNING: model does NOT beat the predict-the-mean baseline")
    print(f"  Predictions outside [{PH_MIN}, {PH_MAX}]: "
          f"{int(np.sum((oof_ph < PH_MIN) | (oof_ph > PH_MAX)))}")

    order = np.argsort(y_ph)[:n_show]
    print(f"\n  {n_show} lowest-pH records (actual vs held-out prediction):")
    print(f"  {'rec_id':<10}{'actual':>8}{'pred':>8}{'error':>8}")
    for i in order:
        print(f"  {str(df_meta.iloc[i]['rec_id']):<10}{y_ph[i]:>8.3f}{oof_ph[i]:>8.3f}{err[i]:>+8.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_ph, oof_ph, s=12, alpha=0.6)
    lo, hi = min(y_ph.min(), oof_ph.min()) - 0.02, max(y_ph.max(), oof_ph.max()) + 0.02
    plt.plot([lo, hi], [lo, hi], "k--", label="identity")
    plt.axhspan(PH_MIN, PH_MAX, color="green", alpha=0.05)
    plt.xlabel("Actual pH")
    plt.ylabel("Predicted pH (out-of-fold)")
    plt.title(f"CNN pH: MAE {mae:.3f}, r {r:.3f}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, "cnn_pred_vs_actual.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"\n  Scatter saved to {path}")


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("        CTG DEEP TEMPORAL SALIENCY ANALYSIS")
    print("=" * 70)

    loader = CTUDataLoader()
    df_meta, fhr_signals, uc_signals = loader.load_dataset(window_minutes=30.0)
    y_ph = df_meta["pH"].values.astype(np.float32)
    y_710 = df_meta["pH_less_710"].values.astype(int)
    y_705 = df_meta["pH_less_705"].values.astype(int)
    print(f"Loaded {len(df_meta)} records.")
    min_ph_idx = int(np.argmin(y_ph))
    sample_rec = df_meta.iloc[min_ph_idx]["rec_id"]
    sample_ph = float(y_ph[min_ph_idx])
    print(f"\nSelected record {sample_rec}, actual pH {sample_ph:.3f}")

    trainer = DeepModelTrainer(
        n_splits=5,
        epochs=20,
        lr=1e-3,
        # pyrefly: ignore[unexpected-keyword]
        verbose_every=1
    )
    oof_ph, _, _, cnn_models = trainer.train_evaluate(
        CTG1DCNN,
        fhr_signals,
        uc_signals,
        y_ph,
        y_710,
        y_705,
        model_name="1D-CNN Saliency Model"
    )

    diagnostics(y_ph, oof_ph, df_meta, results_dir)

    # Use the fold model that did NOT train on this record
    fold_i = [i for i, v in enumerate(trainer.fold_val_indices) if min_ph_idx in v][0]
    cnn_model = cnn_models[fold_i]
    print(f"\nRecord {sample_rec} was held out in fold {fold_i + 1}; using that model.")

    engine = DeepTemporalSaliency(cnn_model)
    pred_ph, fhr_sal, uc_sal = engine.compute_saliency(fhr_signals[min_ph_idx], uc_signals[min_ph_idx])

    print(f"Model predicted pH: {pred_ph:.3f}")
    print(f"Actual pH:          {sample_ph:.3f}")
    print(f"Abs error:          {abs(pred_ph - sample_ph):.3f}")
    if not (PH_MIN <= pred_ph <= PH_MAX):
        print(f"  WARNING: prediction outside physiological range [{PH_MIN}, {PH_MAX}]")

    out_png = os.path.join(results_dir, "deep_temporal_saliency.png")
    engine.plot_temporal_saliency(fhr_signals[min_ph_idx], uc_signals[min_ph_idx],
                                  rec_id=sample_rec, actual_ph=sample_ph, save_path=out_png)

    np.savez_compressed(os.path.join(results_dir, "deep_temporal_saliency_data.npz"),
                        fhr_saliency=fhr_sal, uc_saliency=uc_sal,
                        actual_ph=sample_ph, predicted_ph=pred_ph, record_index=min_ph_idx)
    print(f"\nPlot saved to {out_png}")


if __name__ == "__main__":
    main()