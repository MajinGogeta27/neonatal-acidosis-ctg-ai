import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

class ModelEvaluator:
    """
    Evaluation framework for Continuous pH Regression and Threshold Classification (pH < 7.10 & pH < 7.05).
    Compares Direct Binary Classifiers, Thresholded Regressors, and Categorical CTG Baseline.
    """
    def __init__(self):
        pass

    def evaluate_regression(self, y_true, y_pred, model_name="Model"):
        """
        Calculates regression metrics: MSE, RMSE, MAE, R^2, Pearson r, Spearman rho.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        pearson_r, p_pearson = stats.pearsonr(y_true, y_pred)
        spearman_rho, p_spearman = stats.spearmanr(y_true, y_pred)

        return {
            'Model': model_name,
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'Pearson_r': float(pearson_r),
            'Pearson_p': float(p_pearson),
            'Spearman_rho': float(spearman_rho),
            'Spearman_p': float(p_spearman)
        }

    def evaluate_classification(self, y_true, y_prob, threshold_val=0.5, model_name="Classifier"):
        """
        Calculates classification metrics: AUROC, AUPRC, Sensitivity @ 80%/90% Specificity, F1, Precision, Specificity.
        """
        auroc = roc_auc_score(y_true, y_prob)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall_arr, precision_arr)

        y_pred = (y_prob >= threshold_val).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # Sensitivity at fixed specificities (80% and 90%)
        sens_at_80spec = self._get_sensitivity_at_specificity(y_true, y_prob, target_spec=0.80)
        sens_at_90spec = self._get_sensitivity_at_specificity(y_true, y_prob, target_spec=0.90)

        return {
            'Model': model_name,
            'AUROC': float(auroc),
            'AUPRC': float(auprc),
            'Sensitivity': float(sensitivity),
            'Specificity': float(specificity),
            'Precision': float(precision),
            'F1_Score': float(f1),
            'Sens@80%Spec': float(sens_at_80spec),
            'Sens@90%Spec': float(sens_at_90spec),
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        }

    def _get_sensitivity_at_specificity(self, y_true, y_prob, target_spec=0.80):
        # Scan operating thresholds
        thresholds = np.linspace(0.0, 1.0, 500)
        best_sens = 0.0
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if spec >= target_spec:
                if sens > best_sens:
                    best_sens = sens
        return best_sens

    def compute_categorical_ctg_baseline(self, df_meta, y_710_true, y_705_true):
        """
        Computes performance of standard clinical rule / categorical CTG status quo.
        Rule: Flag as abnormal if deceleration metrics or STV are pathological.
        """
        # Clinical proxy rule: Prolonged II stage, low STV, or tachysystole/deceleration indicator
        # We can construct a combined FIGO risk score baseline
        baseline_score = np.zeros(len(df_meta), dtype=np.float32)
        if 'stv' in df_meta.columns:
            baseline_score += (df_meta['stv'] < 3.0).astype(float) * 2.0
        if 'decel_count' in df_meta.columns:
            baseline_score += (df_meta['decel_count'] > 3).astype(float) * 1.5
        if 'Stage_II_duration' in df_meta.columns:
            baseline_score += (df_meta['Stage_II_duration'] > 45).astype(float) * 1.0

        # Normalize score to [0, 1] probability proxy
        max_s = np.max(baseline_score)
        if max_s > 0:
            baseline_probs = baseline_score / max_s
        else:
            baseline_probs = np.random.uniform(0.1, 0.3, size=len(df_meta))

        metrics_710 = self.evaluate_classification(y_710_true, baseline_probs, threshold_val=0.5, model_name="Categorical CTG (FIGO Clinical Rule)")
        metrics_705 = self.evaluate_classification(y_705_true, baseline_probs, threshold_val=0.5, model_name="Categorical CTG (FIGO Clinical Rule)")

        return metrics_710, metrics_705

    def plot_regression_scatter(self, y_true, y_pred_dict, save_path="results/regression_scatter.png"):
        """
        Plots Predicted pH vs Actual Cord Blood pH scatter plots for all regression models,
        ordered left-to-right from highest to lowest R².
        """
        # Sort by R² descending (best model first / leftmost panel)
        sorted_items = sorted(
            y_pred_dict.items(),
            key=lambda kv: r2_score(y_true, kv[1]),
            reverse=True,
        )

        n_models = len(sorted_items)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
        axes = axes.flatten()

        for idx, (name, y_pred) in enumerate(sorted_items):
            ax = axes[idx]
            sns.regplot(x=y_true, y=y_pred, ax=ax, scatter_kws={'alpha': 0.6, 'color': '#1f77b4'}, line_kws={'color': '#d62728'})
            ax.axhline(7.10, color='gray', linestyle='--', label='pH = 7.10')
            ax.axvline(7.10, color='gray', linestyle='--')
            ax.set_title(f"{name}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Actual Cord Blood pH", fontsize=10)
            ax.set_ylabel("Predicted pH", fontsize=10)

            r2 = r2_score(y_true, y_pred)
            r, _ = stats.pearsonr(y_true, y_pred)
            ax.annotate(f"$R^2$ = {r2:.3f}\n$r$ = {r:.3f}", xy=(0.05, 0.82), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
            ax.legend(loc='lower right')
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved regression scatter plot to {save_path}")

    def bootstrap_auroc_ci(self, y_true, y_prob, n_bootstrap: int = 2000,
                           alpha: float = 0.05, random_state: int = 42):
        """
        Compute bootstrap 95% CI for AUROC.

        Returns
        -------
        (auroc, ci_lower, ci_upper) : tuple of float
        """
        rng = np.random.default_rng(random_state)
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        auroc = roc_auc_score(y_true, y_prob)

        boot_aurocs = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            boot_aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))

        boot_aurocs = np.array(boot_aurocs)
        ci_lower = float(np.percentile(boot_aurocs, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))
        return float(auroc), ci_lower, ci_upper

    def plot_roc_curves(self, y_true, prob_dict,
                        title: str = "ROC Curves (pH < 7.10)",
                        save_path: str = "results/roc_curves.png",
                        n_bootstrap: int = 2000):
        """
        Plots ROC curves with 95% bootstrap confidence-interval bands.

        Parameters
        ----------
        y_true : array-like of int
            Binary ground-truth labels.
        prob_dict : dict[str, array-like]
            Mapping from model name to predicted probabilities.
        title : str
            Plot title.
        save_path : str
            Output file path.
        n_bootstrap : int
            Number of bootstrap resamples for CI estimation (0 = no CI bands).
        """
        from sklearn.metrics import roc_curve

        COLORS = [
            '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf',
        ]

        # Sort models by AUROC descending so legend reads best → worst
        sorted_items = sorted(
            prob_dict.items(),
            key=lambda kv: roc_auc_score(y_true, np.asarray(kv[1])),
            reverse=True,
        )

        fig, ax = plt.subplots(figsize=(8, 7))

        for idx, (name, probs) in enumerate(sorted_items):
            color = COLORS[idx % len(COLORS)]
            probs = np.asarray(probs)

            fpr, tpr, _ = roc_curve(y_true, probs)

            if n_bootstrap > 0:
                auroc, ci_lo, ci_hi = self.bootstrap_auroc_ci(
                    y_true, probs, n_bootstrap=n_bootstrap)
                label = (f"{name}\nAUROC = {auroc:.3f} "
                         f"(95% CI {ci_lo:.3f}–{ci_hi:.3f})")

                # Bootstrap CI band
                rng = np.random.default_rng(42)
                n = len(y_true)
                mean_fpr = np.linspace(0, 1, 200)
                tprs_boot = []
                for _ in range(n_bootstrap):
                    bi = rng.integers(0, n, size=n)
                    if len(np.unique(np.asarray(y_true)[bi])) < 2:
                        continue
                    bfpr, btpr, _ = roc_curve(
                        np.asarray(y_true)[bi], probs[bi])
                    tprs_boot.append(np.interp(mean_fpr, bfpr, btpr))

                tprs_boot = np.array(tprs_boot)
                tpr_lo = np.percentile(tprs_boot, 2.5, axis=0)
                tpr_hi = np.percentile(tprs_boot, 97.5, axis=0)
                ax.fill_between(mean_fpr, tpr_lo, tpr_hi,
                                alpha=0.12, color=color)
            else:
                auroc = roc_auc_score(y_true, probs)
                label = f"{name} (AUROC = {auroc:.3f})"

            ax.plot(fpr, tpr, color=color, lw=2.0, label=label)

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUROC = 0.500)')
        ax.set_xlabel('False Positive Rate  (1 – Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved ROC curve to {save_path}")


    def plot_confusion_matrices(self, y_true, prob_dict,
                                threshold_val: float = 0.5,
                                top_n: int = 6,
                                title: str = "Confusion Matrices (pH < 7.10)",
                                class_labels: list = None,
                                save_path: str = "results/confusion_matrices.png"):
        """
        Plot a grid of styled confusion matrices (purple palette) for the top-N
        models sorted by AUROC, best → worst.

        Colour scheme
        -------------
        • Dark violet  (#6C3483)  → Correct predictions  (TP / TN)
        • Light lavender (#D7BDE2) → Errors              (FP / FN)

        A shared colour legend is drawn below the figure grid.

        Each cell shows
        ---------------
        • Cell type label   (TP / TN / FP / FN) – top-left corner
        • Absolute count    – large, bold, centred
        • Row-%             – smaller, below count

        A per-panel metric strip below each subplot shows
        Sensitivity · Specificity · PPV · NPV · F1.
        """
        import matplotlib.patches as mpatches

        if class_labels is None:
            class_labels = ["Normal pH", "Acidemia"]

        # ── Colour palette ────────────────────────────────────────────
        _CORRECT = "#6C3483"   # dark violet  (correct: TP, TN)
        _ERROR   = "#D7BDE2"   # light lavender (errors: FP, FN)
        _TXT_ON_DARK  = "#FFFFFF"
        _TXT_ON_LIGHT = "#4A235A"   # deep purple text on lavender bg

        # Cell-type labels  [row=true, col=pred]
        _CELL_LABEL = [["TN", "FP"],
                       ["FN", "TP"]]

        # Sort by AUROC descending, keep top_n
        scored = sorted(
            prob_dict.items(),
            key=lambda kv: roc_auc_score(y_true, np.asarray(kv[1])),
            reverse=True,
        )[:top_n]

        n = len(scored)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))

        # Extra bottom space for the shared legend
        fig = plt.figure(figsize=(5.8 * ncols, 5.6 * nrows + 0.9))
        gs  = fig.add_gridspec(
            nrows, ncols,
            top=0.93, bottom=0.14,
            left=0.06, right=0.97,
            hspace=0.55, wspace=0.35,
        )
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98,
                     color='#2C0A37')

        axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)]
                for r in range(nrows)]

        for panel_idx, (name, probs) in enumerate(scored):
            row_idx, col_idx = divmod(panel_idx, ncols)
            ax = axes[row_idx][col_idx]

            probs  = np.asarray(probs)
            y_pred = (probs >= threshold_val).astype(int)
            cm     = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp_val = cm.ravel()

            # Row-normalised percentages
            cm_norm  = cm.astype(float)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct   = cm_norm / row_sums * 100.0

            bg_colors = [
                [_CORRECT, _ERROR],
                [_ERROR,   _CORRECT],
            ]
            txt_colors = [
                [_TXT_ON_DARK,  _TXT_ON_LIGHT],
                [_TXT_ON_LIGHT, _TXT_ON_DARK],
            ]

            for i in range(2):
                for j in range(2):
                    # Background rectangle
                    ax.add_patch(plt.Rectangle(
                        (j, 1 - i), 1, 1,
                        color=bg_colors[i][j], alpha=0.88, zorder=0,
                        transform=ax.transData,
                    ))

                    count = int(cm[i, j])
                    pct   = cm_pct[i, j]
                    tc    = txt_colors[i][j]
                    cell  = _CELL_LABEL[i][j]

                    # Cell-type label (top-left corner of cell)
                    ax.text(j + 0.08, 1 - i + 0.88, cell,
                            ha='left', va='top',
                            fontsize=9, fontweight='bold',
                            color=tc, alpha=0.85, zorder=3)

                    # Count
                    ax.text(j + 0.5, 1 - i + 0.57, f"{count}",
                            ha='center', va='center',
                            fontsize=24, fontweight='bold',
                            color=tc, zorder=3)

                    # Row %
                    ax.text(j + 0.5, 1 - i + 0.30, f"({pct:.1f}%)",
                            ha='center', va='center',
                            fontsize=11, color=tc, zorder=3)

            # White grid dividers
            ax.axhline(1, color='white', lw=2.5, zorder=2)
            ax.axvline(1, color='white', lw=2.5, zorder=2)

            # Axis formatting
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_yticks([0.5, 1.5])
            ax.set_xticklabels(class_labels, fontsize=9.5, color='#2C0A37')
            ax.set_yticklabels(class_labels[::-1], fontsize=9.5, color='#2C0A37')
            ax.set_xlabel("Predicted Label", fontsize=10.5, labelpad=5,
                          color='#2C0A37')
            ax.set_ylabel("True Label", fontsize=10.5, labelpad=5,
                          color='#2C0A37')
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_edgecolor('#A569BD')
                spine.set_linewidth(1.2)

            # Panel title
            auroc = roc_auc_score(y_true, probs)
            display_name = name if len(name) <= 40 else name[:38] + "…"
            ax.set_title(f"{display_name}\nAUROC = {auroc:.3f}",
                         fontsize=9.5, fontweight='bold', pad=7,
                         color='#2C0A37')

            # Per-panel metric strip
            sensitivity = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0.0
            specificity = tn / (tn + fp)         if (tn + fp)     > 0 else 0.0
            ppv = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0.0
            npv = tn / (tn + fn)         if (tn + fn)     > 0 else 0.0
            f1  = (2 * ppv * sensitivity / (ppv + sensitivity)
                   if (ppv + sensitivity) > 0 else 0.0)

            ax.text(
                0.5, -0.18,
                (f"Sens {sensitivity:.2f}  |  Spec {specificity:.2f}  |  "
                 f"PPV {ppv:.2f}  |  NPV {npv:.2f}  |  F1 {f1:.2f}"),
                transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color='#4A235A',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='#F5EEF8', ec='#A569BD', lw=0.8),
            )

        # Hide unused panels
        for empty_idx in range(n, nrows * ncols):
            r, c = divmod(empty_idx, ncols)
            axes[r][c].set_visible(False)

        # ── Shared colour legend ──────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=_CORRECT, alpha=0.88,
                           label="Correct prediction  (TP = True Positive / TN = True Negative)"),
            mpatches.Patch(color=_ERROR,   alpha=0.88,
                           label="Prediction error     (FP = False Positive / FN = False Negative)"),
        ]
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=1,
            fontsize=9.5,
            frameon=True,
            framealpha=0.95,
            edgecolor='#A569BD',
            facecolor='#FAF0FF',
            title="Colour Legend",
            title_fontsize=10,
            bbox_to_anchor=(0.5, 0.01),
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved confusion matrices to {save_path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    y_true = np.random.normal(7.22, 0.1, 100)
    y_pred = y_true + np.random.normal(0, 0.05, 100)
    reg_res = evaluator.evaluate_regression(y_true, y_pred, "Test Model")
    print("Regression test metrics:", reg_res)
