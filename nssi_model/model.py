"""Preprocessing, five experts, OOF stacking, and prediction."""
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class SelectColumns(TransformerMixin, BaseEstimator):
    def __init__(self, prefix=None, columns=None):
        self.prefix = prefix
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns) if self.columns is not None else [c for c in X if c.startswith(self.prefix)]
        if not self.columns_ or not set(self.columns_).issubset(X.columns):
            raise ValueError(f"Missing columns for {self.prefix or self.columns}")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        return X.loc[:, self.columns_].copy()


class IntervalCodes(TransformerMixin, BaseEstimator):
    """S3: 'max' -> 1.1 * training max; 'min' -> 0.9 * training min.

    Unknown text is rejected, never silently interpreted as zero or missing.
    Numeric < or > strings need an explicit source-data decision before import.
    """
    def fit(self, X, y=None):
        a = self._frame(X)
        numeric = a.mask(a.isin(["max", "min"])).astype(float)
        self.low_ = numeric.min().to_numpy() * 0.9
        self.high_ = numeric.max().to_numpy() * 1.1
        if np.any(~np.isfinite(self.low_)) or np.any(~np.isfinite(self.high_)):
            raise ValueError("A training feature has no finite numeric observations; no imputation rule is specified.")
        self.n_features_in_ = a.shape[1]
        return self

    @staticmethod
    def _frame(X):
        a = pd.DataFrame(X).reset_index(drop=True).copy()
        a.columns = range(a.shape[1])
        return a

    def transform(self, X):
        check_is_fitted(self, "low_")
        a = self._frame(X)
        if a.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        for j in a:
            a[j] = a[j].replace({"min": self.low_[j], "max": self.high_[j]})
        result = a.astype(float).to_numpy()
        if np.isinf(result).any():
            raise ValueError("Infinite measurement")
        return result


def numeric_pipeline():
    return Pipeline([("intervals", IntervalCodes()),
                     ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler())])


CLINICAL_NUM = ("clinical__age", "clinical__bdi_ii", "clinical__scared")


CLINICAL_CAT = ("clinical__sex", "clinical__depression", "clinical__anxiety")


def clinical_transformer():
    # Categories are a fixed input schema; Unknown columns exist in every fold.
    cats = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(categories=[["Female", "Male", "Unknown"],
                                               ["No", "Yes", "Unknown"],
                                               ["No", "Yes", "Unknown"]],
                                 handle_unknown="error", sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric_pipeline(), list(CLINICAL_NUM)),
                              ("categorical", cats, list(CLINICAL_CAT))])


class ConvergedLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                return super().fit(X, y, sample_weight=sample_weight)
            except ConvergenceWarning as exc:
                raise RuntimeError(
                    "Logistic fit did not converge; no candidate score or prediction "
                    f"is accepted (solver={self.solver}, C={self.C}, "
                    f"l1_ratio={self.l1_ratio}, max_iter={self.max_iter}, "
                    f"training_shape={X.shape})."
                ) from exc


class RobustVotingSelector(TransformerMixin, BaseEstimator):
    def __init__(self, min_votes=2, random_state=42):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.p_values_ = np.array([mannwhitneyu(X[y == 0, j], X[y == 1, j],
                                               alternative="two-sided").pvalue
                                   for j in range(X.shape[1])])
        en = ConvergedLogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                C=1.0, class_weight="balanced", max_iter=100000,
                                random_state=self.random_state).fit(X, y)
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    n_jobs=1, random_state=self.random_state).fit(X, y)
        self.en_support_ = en.coef_[0] != 0
        self.importances_ = rf.feature_importances_
        # Exact ceil(p/4) membership, ties broken by original column order.
        k = max(1, int(np.ceil(X.shape[1] * 0.25)))
        self.rf_support_ = np.zeros(X.shape[1], dtype=bool)
        self.rf_support_[np.argsort(-self.importances_, kind="stable")[:k]] = True
        self.votes_ = (self.p_values_ < 0.05).astype(int) + self.en_support_.astype(int) + self.rf_support_.astype(int)
        self.mask_ = self.votes_ >= self.min_votes
        if not self.mask_.any():
            raise ValueError("No feature receives two votes; S4 specifies no fallback. Do not substitute all features.")
        return self

    def transform(self, X):
        check_is_fitted(self, "mask_")
        return np.asarray(X)[:, self.mask_]

    def get_support(self):
        check_is_fitted(self, "mask_")
        return self.mask_.copy()


GENETIC = ("PRS_EUR", "Rare_Burden")


def logistic(max_iter=100):
    return ConvergedLogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=max_iter,
                              tol=1e-4, class_weight="balanced", random_state=42)


def experts():
    result = []
    for modality, prefix in [("Met", "met__"), ("Inf", "inf__")]:
        # Tune the entire pipeline, so even internal tuning validation rows do
        # not contribute to fitted preprocessing. Same C grid/l1 ratios as before.
        en = GridSearchCV(Pipeline([
            ("preprocess", numeric_pipeline()),
            ("model", ConvergedLogisticRegression(penalty="elasticnet", solver="saga",
                                          class_weight="balanced", max_iter=100000,
                                          random_state=42))]),
            param_grid={"model__C": np.logspace(-4, 4, 10), "model__l1_ratio": [0.5, 0.7]},
            cv=StratifiedKFold(5, shuffle=False), scoring="accuracy", n_jobs=1,
            refit=True, error_score="raise", return_train_score=False)
        result.append((modality + "_EN", Pipeline([
            ("columns", SelectColumns(prefix=prefix)), ("search", en)])))
        result.append((modality + "_RF", Pipeline([
            ("columns", SelectColumns(prefix=prefix)), ("preprocess", numeric_pipeline()),
            ("screen", RobustVotingSelector()),
            ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                               random_state=42, n_jobs=1))])))
    result.append(("Blood_RF", Pipeline([
        ("columns", SelectColumns(prefix="blood__")), ("preprocess", numeric_pipeline()),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=5,
                                         class_weight="balanced", random_state=42, n_jobs=1))])))
    return result


def clinical_model():
    return Pipeline([("preprocess", clinical_transformer()), ("model", logistic(1000))])


def genomic_model(random_state=42):
    # Recovered from the later Model__main.py source snapshot (2026-10-15), lines 195-201.
    return Pipeline([("columns", SelectColumns(columns=GENETIC)),
                     ("preprocess", numeric_pipeline()),
                     ("model", ConvergedLogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                                   solver="lbfgs", max_iter=1000, random_state=random_state))])


PANEL_MARKERS = ("CXCL5", "IL24", "ADA", "TWEAK", "NT-3", "D-serine", "cortisol/DHEA-S ratio")


def panel_model(columns):
    if not isinstance(columns, dict) or set(columns) != set(PANEL_MARKERS):
        raise ValueError("Provide a marker-to-column mapping with exactly the seven S9 marker names")
    columns = [columns[marker] for marker in PANEL_MARKERS]
    if len(set(columns)) != 7 or any(not isinstance(c, str) for c in columns):
        raise ValueError("Each S9 marker must map to a distinct input column")
    return Pipeline([("columns", SelectColumns(columns=tuple(columns))),
                     ("preprocess", numeric_pipeline()), ("model", logistic())])


class UniversalStacking(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimators, meta_classifier, cv=5, use_genomics=True,
                 use_interaction=True, genomic_columns=GENETIC, random_state=42):
        self.base_estimators = base_estimators
        self.meta_classifier = meta_classifier
        self.cv = cv
        self.use_genomics = use_genomics
        self.use_interaction = use_interaction
        self.genomic_columns = genomic_columns
        self.random_state = random_state

    def _meta(self, probabilities, genetics):
        if not self.use_genomics:
            return probabilities
        blocks = [probabilities, genetics]
        if self.use_interaction:
            blocks.append((probabilities[:, :, None] * genetics[:, None, :]).reshape(len(probabilities), -1))
        return np.column_stack(blocks)

    def fit(self, X, y):
        if hasattr(y, "index") and not X.index.equals(y.index):
            raise ValueError("X and y participant order differs")
        if not X.columns.is_unique or X.index.isna().any():
            raise ValueError("Feature names must be unique and participant IDs nonmissing")
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X/y lengths differ")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Labels must be 0=non-NSSI and 1=recent-NSSI")
        if not X.index.is_unique:
            raise ValueError("Duplicate participant identifiers")
        self.base_names_ = [name for name, _ in self.base_estimators]
        if not self.base_names_ or len(set(self.base_names_)) != len(self.base_names_):
            raise ValueError("Experts must have unique names")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.oof_probabilities_ = np.full((len(X), len(self.base_estimators)), np.nan)
        self.oof_genetics_ = np.full((len(X), len(self.genomic_columns)), np.nan) if self.use_genomics else None
        self.fold_audit_ = []
        self.genetic_fold_transforms_ = []
        self.oof_fold_ = np.full(len(X), -1, dtype=int)
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            self.oof_fold_[va] = fold
            self.fold_audit_.append({"fold": fold, "train": X.index[tr].tolist(), "validation": X.index[va].tolist()})
            for j, (_, estimator) in enumerate(self.base_estimators):
                fitted = clone(estimator).fit(X.iloc[tr], y[tr])
                self.oof_probabilities_[va, j] = fitted.predict_proba(X.iloc[va])[:, 1]
            if self.use_genomics:
                transform = numeric_pipeline().fit(X.iloc[tr].loc[:, list(self.genomic_columns)])
                self.oof_genetics_[va] = transform.transform(X.iloc[va].loc[:, list(self.genomic_columns)])
                self.genetic_fold_transforms_.append(transform)
        self.oof_meta_features_ = self._meta(self.oof_probabilities_, self.oof_genetics_)
        if not np.isfinite(self.oof_meta_features_).all():
            raise ValueError("Nonfinite OOF meta inputs")
        self.meta_feature_names_ = self.base_names_.copy()
        if self.use_genomics:
            self.meta_feature_names_ += list(self.genomic_columns)
            if self.use_interaction:
                self.meta_feature_names_ += [f"{b}_x_{g}" for b in self.base_names_ for g in self.genomic_columns]
        self.fitted_meta_model_ = clone(self.meta_classifier).fit(self.oof_meta_features_, y)
        self.fitted_base_models_ = [clone(est).fit(X, y) for _, est in self.base_estimators]
        if self.use_genomics:
            self.genome_transform_ = numeric_pipeline().fit(X.loc[:, list(self.genomic_columns)])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_meta_model_")
        base = np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_base_models_])
        genetics = self.genome_transform_.transform(X.loc[:, list(self.genomic_columns)]) if self.use_genomics else None
        return self.fitted_meta_model_.predict_proba(self._meta(base, genetics))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def required_columns(model):
    """Columns consumed by fitted experts, excluding unused training metadata."""
    required = set()
    for expert in model.fitted_base_models_:
        steps = expert.named_steps
        if "columns" in steps:
            required.update(steps["columns"].columns_)
        elif "preprocess" in steps and hasattr(steps["preprocess"], "transformers_"):
            for _, transformer, columns in steps["preprocess"].transformers_:
                if isinstance(transformer, str) and transformer == "drop":
                    continue
                required.update(columns)
        else:
            raise ValueError("Cannot determine required columns for this model artifact")
    if model.use_genomics:
        required.update(model.genomic_columns)
    return required


def main():
    from .data import load_schema, adapt_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted local joblib artifact")
    parser.add_argument("--data", required=True, help="CSV with caller-defined columns")
    parser.add_argument("--schema", required=True, help="Local column mapping retaining training feature aliases")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    model = joblib.load(args.model)
    X, _, _ = adapt_inputs(args.data, load_schema(args.schema), outcome=False, required=required_columns(model))
    p = model.predict_proba(X)[:, 1]
    pd.DataFrame({"sample_id": X.index, "probability": p, "prediction_at_0_5": (p >= 0.5).astype(int)}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
