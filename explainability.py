import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from models_deep import prepare_input, valid_length_mask, PH_MIN, PH_MAX

class FeatureExplainability:
    """
    Computes SHAP feature importance and feature attribution rankings
    for engineered CTG baseline models.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def compute_shap_values(
        self,
        model,
        X_sample,
        save_plot_path="results/shap_summary.png"
    ):
        """
        Computes SHAP values using shap.Explainer.
        Falls back to tree feature importances or linear coefficients.
        """

        try:
            import shap

            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

            plt.figure(figsize=(10, 6))

            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                show=False
            )

            plt.tight_layout()

            plt.savefig(
                save_plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Saved SHAP summary plot to {save_plot_path}"
            )

            return shap_values

        except Exception as e:

            print(
                f"SHAP explainer fallback to tree feature "
                f"importances: {e}"
            )

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)

            else:
                importances = (
                    np.ones(len(self.feature_names))
                    / len(self.feature_names)
                )

            df_imp = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances
            })

            df_imp = df_imp.sort_values(
                by="importance",
                ascending=False
            )

            plt.figure(figsize=(10, 6))

            plt.barh(
                df_imp["feature"].head(15)[::-1],
                df_imp["importance"].head(15)[::-1]
            )

            plt.xlabel(
                "Feature Importance",
                fontsize=11
            )

            plt.title(
                "Top 15 Engineered Features Driving pH Prediction",
                fontsize=12,
                fontweight="bold"
            )

            plt.tight_layout()

            plt.savefig(
                save_plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Saved feature importances plot to "
                f"{save_plot_path}"
            )

            return df_imp


class DeepTemporalSaliency:
    """
    SmoothGrad temporal saliency for the CTG CNN.
    Uses the SAME preprocessing as training (prepare_input).
    """

    def __init__(self, model, device=None, n_smooth=25, noise_std=0.1,
                 smooth_seconds=10.0, fs=4.0):
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_smooth = n_smooth
        self.noise_std = noise_std
        self.smooth_len = max(1, int(smooth_seconds * fs))
        self.fs = fs
        self.model.to(self.device)
        self.model.eval()

    def compute_saliency(self, fhr_single, uc_single):
        x = prepare_input(fhr_single, uc_single)          # [3, L], normalized + masked
        mask = x[2]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = x_t[:, 2:3, :]

        with torch.no_grad():
            pred_ph = float(self.model(x_t)[0].reshape(-1)[0].item())

        grads = np.zeros((2, x.shape[1]), dtype=np.float32)
        for _ in range(self.n_smooth):
            noise = torch.randn_like(x_t[:, :2]) * self.noise_std * mask_t
            noisy = x_t.clone()
            noisy[:, :2] = noisy[:, :2] + noise
            noisy.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            self.model(noisy)[0].sum().backward()
            grads += noisy.grad[0, :2].abs().cpu().numpy()
        grads /= self.n_smooth

        kernel = np.ones(self.smooth_len, dtype=np.float32) / self.smooth_len

        def finish(g):
            g = np.convolve(g, kernel, mode="same") * mask
            valid = g[mask > 0]
            scale = np.percentile(valid, 99) if valid.size else 0.0
            return np.clip(g / scale, 0, 1) if scale > 0 else g

        return pred_ph, finish(grads[0]), finish(grads[1])

    def plot_temporal_saliency(self, fhr_single, uc_single, rec_id, actual_ph,
                               save_path="results/deep_temporal_saliency.png"):
        pred_ph, fhr_sal, uc_sal = self.compute_saliency(fhr_single, uc_single)

        fhr = np.nan_to_num(np.asarray(fhr_single, dtype=np.float32), nan=140.0)
        uc = np.nan_to_num(np.asarray(uc_single, dtype=np.float32), nan=20.0)
        n = len(fhr)
        t = np.arange(n) / self.fs / 60.0
        n_valid = int(valid_length_mask(fhr, uc).sum())

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
        im = None
        panels = [
            (axes[0], fhr, fhr_sal, "Fetal Heart Rate (bpm)", "FHR Temporal Saliency"),
            (axes[1], uc, uc_sal, "Uterine Activity", "Uterine Activity Temporal Saliency"),
        ]
        for ax, sig, sal, ylabel, title in panels:
            lo, hi = float(sig.min()), float(sig.max())
            pad = 0.05 * max(hi - lo, 1e-3)
            im = ax.imshow(sal[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1,
                           alpha=0.65, origin="lower",
                           extent=[t[0], t[-1], lo - pad, hi + pad], zorder=1)
            ax.plot(t, sig, color="black", linewidth=1, zorder=3)
            if n_valid < n - 1:
                ax.axvspan(t[n_valid - 1], t[-1], color="gray", alpha=0.2, zorder=2,
                           label="padded / no signal")
                ax.legend(loc="upper right")
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        axes[1].set_xlabel("Time (minutes)")
        fig.colorbar(im, ax=axes, label="Normalized saliency", pad=0.01)

        fig.suptitle(
            f"CTG Temporal Saliency: Record {rec_id}\n"
            f"Actual pH: {actual_ph:.3f} | Predicted pH: {pred_ph:.3f} | "
            f"|Error|: {abs(pred_ph - actual_ph):.3f}",
            fontsize=14,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved temporal saliency plot to {save_path}")
        return pred_ph, fhr_sal, uc_sal


if __name__ == "__main__":

    print("Explainability module loaded successfully.")