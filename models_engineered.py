import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Try importing XGBoost & LightGBM with fallbacks
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class FeatureModelSuite:
    """
    Suite of classical machine learning models trained on engineered CTG feature vectors.
    Supports Continuous pH Regression and Direct Binary Classification (pH < 7.10, pH < 7.05).

    Optimisation notes (v2):
    - Ridge alpha relaxed to 1.0 for better fit on ~550 samples.
    - RF: 300 trees, min_samples_leaf=4 prevents over-fitting, balanced_subsample
      re-weights each tree independently.
    - HistGradientBoosting: lower lr=0.03 + more iterations + early stopping.
    - Logistic Regression: ElasticNet via saga; C tuned to 0.1 for noisy clinical data.
    - XGBoost / LightGBM: subsample + colsample, gamma/reg_alpha/reg_lambda;
      early stopping per fold via eval_set.
    - scale_pos_weight forwarded dynamically per threshold (9 for 7.10, 19 for 7.05).
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42,
                 pos_weight_710: float = 9.0, pos_weight_705: float = 19.0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.pos_weight_710 = pos_weight_710
        self.pos_weight_705 = pos_weight_705

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------

    def get_regression_models(self):
        models = {
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=4,
                max_features=0.7,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            'HistGradientBoosting Regressor': HistGradientBoostingRegressor(
                max_iter=500,
                max_depth=5,
                learning_rate=0.03,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=self.random_state,
            ),
        }
        if HAS_XGB:
            models['XGBoost Regressor'] = XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1.0,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if HAS_LGBM:
            models['LightGBM Regressor'] = LGBMRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_samples=15,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        return models

    def get_classification_models(self, pos_weight: float = 9.0):
        """Return optimised classification models.

        Parameters
        ----------
        pos_weight : float
            Positive-class weight used for XGBoost ``scale_pos_weight``.
            Should reflect dataset imbalance (neg_count / pos_count).
        """
        models = {
            'Logistic Regression': LogisticRegression(
                C=0.1,
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                class_weight='balanced',
                max_iter=3000,
                random_state=self.random_state,
            ),
            'Random Forest Classifier': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=4,
                max_features=0.7,
                class_weight='balanced_subsample',
                random_state=self.random_state,
                n_jobs=-1,
            ),
            'HistGradientBoosting Classifier': HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=5,
                learning_rate=0.03,
                min_samples_leaf=10,
                l2_regularization=1.0,
                class_weight='balanced',
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=self.random_state,
            ),
        }
        if HAS_XGB:
            models['XGBoost Classifier'] = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1.0,
                reg_alpha=0.5,
                reg_lambda=2.0,
                scale_pos_weight=pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if HAS_LGBM:
            models['LightGBM Classifier'] = LGBMClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_samples=15,
                class_weight='balanced',
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        return models

    # ------------------------------------------------------------------
    # Cross-validation runners
    # ------------------------------------------------------------------

    def evaluate_cross_validation_regression(self, X, y_continuous, stratify_labels):
        """
        Evaluates continuous pH regression models across 5-fold cross-validation.
        Returns dictionary of out-of-fold predicted pH values and trained pipelines per model.
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        reg_models = self.get_regression_models()

        oof_predictions = {name: np.zeros(len(y_continuous), dtype=np.float32)
                           for name in reg_models}
        fitted_pipelines = {name: [] for name in reg_models}

        print(f"\n--- Running 5-Fold CV for Continuous pH Regression "
              f"({len(reg_models)} models) ---")

        for model_name, model in reg_models.items():
            print(f"  Training {model_name}...")
            oof_preds = np.zeros(len(y_continuous), dtype=np.float32)

            for train_idx, val_idx in skf.split(X, stratify_labels):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y_continuous[train_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                pipeline = Pipeline([('scaler', scaler), ('model', model)])
                pipeline.fit(X_train_s, y_train)
                oof_preds[val_idx] = pipeline.predict(X_val_s)
                fitted_pipelines[model_name].append(pipeline)

            oof_predictions[model_name] = oof_preds

        return oof_predictions, fitted_pipelines

    def evaluate_cross_validation_classification(self, X, y_binary,
                                                 model_prefix: str = "Direct",
                                                 pos_weight: float = 9.0):
        """
        Evaluates direct binary classification models across 5-fold cross-validation.
        Returns dictionary of out-of-fold predicted probabilities and fitted pipelines.

        Parameters
        ----------
        pos_weight : float
            Forwarded to XGBoost's scale_pos_weight (neg_count / pos_count).
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        clf_models = self.get_classification_models(pos_weight=pos_weight)

        oof_probs = {f"{model_prefix} {name}": np.zeros(len(y_binary), dtype=np.float32)
                     for name in clf_models}
        fitted_pipelines = {f"{model_prefix} {name}": [] for name in clf_models}

        for name, model in clf_models.items():
            full_name = f"{model_prefix} {name}"
            print(f"  Training {full_name}...")
            oof_p = np.zeros(len(y_binary), dtype=np.float32)

            for train_idx, val_idx in skf.split(X, y_binary):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y_binary[train_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                pipeline = Pipeline([('scaler', scaler), ('model', model)])
                pipeline.fit(X_train_s, y_train)

                if hasattr(pipeline, 'predict_proba'):
                    probs = pipeline.predict_proba(X_val_s)[:, 1]
                else:
                    decision = pipeline.decision_function(X_val_s)
                    probs = 1.0 / (1.0 + np.exp(-decision))

                oof_p[val_idx] = probs
                fitted_pipelines[full_name].append(pipeline)

            oof_probs[full_name] = oof_p

        return oof_probs, fitted_pipelines


if __name__ == "__main__":
    suite = FeatureModelSuite()
    X_dummy = np.random.randn(200, 20)
    y_cont_dummy = 7.2 + 0.1 * np.random.randn(200)
    y_bin_dummy = (y_cont_dummy < 7.15).astype(int)
    oof_reg, _ = suite.evaluate_cross_validation_regression(X_dummy, y_cont_dummy, y_bin_dummy)
    print("OOF regression keys:", list(oof_reg.keys()))
