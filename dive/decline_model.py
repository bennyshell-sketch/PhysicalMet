"""Decline classification from the B/T ratio.

Primary model: L2-penalised logistic regression,
``declined ~ log(B/T) + age + sex`` fitted on the n=100 case-control substudy.

Three things make this honest rather than merely fitted:

**Case-control correction.** The substudy sampled 50 decliners and 50 stable
participants. Raw predicted probabilities therefore describe a 50/50 world
that does not exist. A King & Zeng prior correction shifts the intercept back
onto the observed cohort prevalence, and every uncorrected probability is
labelled as such.

**A permutation test on the AUC.** With n=100 and three predictors, a
cross-validated AUC of 0.60 is entirely achievable from noise. The permutation
p-value is reported next to every AUC, and the verdict is driven by the
p-value, not the AUC.

**Conformal calibration.** Individual risk estimates carry a distribution-free
prediction set, and the model abstains when the person sits outside the
covariate region it was trained on.

The gradient-boosting model is a *sensitivity check only*. It is never the
headline result and the module refuses to report it as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SETTINGS, Settings
from .guardrails import ExtrapolationGuard, Prediction, check_training_adequacy

__all__ = [
    "DeclineModel",
    "DeclineModelResult",
    "fit_decline_model",
    "prior_correction_offset",
]

DEFAULT_FEATURES = ("log_BT_ratio", "age_years", "sex")


# --------------------------------------------------------------------------
# Case-control correction
# --------------------------------------------------------------------------

def prior_correction_offset(sample_prevalence: float, population_prevalence: float) -> float:
    """King & Zeng prior correction for the logistic intercept.

    Returns the amount to SUBTRACT from the fitted intercept so that predicted
    probabilities refer to a population with ``population_prevalence`` rather
    than to the artificial ``sample_prevalence`` of the case-control design.
    """
    if not (0 < sample_prevalence < 1) or not (0 < population_prevalence < 1):
        return 0.0
    return float(
        np.log(
            ((1 - population_prevalence) / population_prevalence)
            * (sample_prevalence / (1 - sample_prevalence))
        )
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class DeclineModelResult:
    coefficients: pd.DataFrame
    auc_cv: float
    auc_cv_sd: float
    auc_permutation_p: float
    brier: float
    calibration: pd.DataFrame
    n: int
    n_cases: int
    sample_prevalence: float
    population_prevalence: float | None
    intercept_offset: float
    cv_scheme: str
    verdict: str
    technical_note: str
    sensitivity: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_predictive(self) -> bool:
        """Predictive only if the AUC beats chance by permutation test."""
        return bool(np.isfinite(self.auc_permutation_p) and self.auc_permutation_p < 0.05)

    def coefficient_table(self) -> pd.DataFrame:
        table = self.coefficients.copy()
        table["Odds ratio (95% CI)"] = [
            f"{row['odds_ratio']:.3f} ({row['or_lower']:.3f}-{row['or_upper']:.3f})"
            for _, row in table.iterrows()
        ]
        table["p"] = table["p_value"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
        return table[["term", "coefficient", "Odds ratio (95% CI)", "p"]].rename(
            columns={"term": "Predictor", "coefficient": "log-odds"}
        )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class DeclineModel:
    """L2 logistic regression with case-control awareness and abstention."""

    def __init__(
        self,
        features: Sequence[str] = DEFAULT_FEATURES,
        settings: Settings | None = None,
        population_prevalence: float | None = None,
    ):
        self.cfg = settings or SETTINGS
        self.features = list(features)
        self.population_prevalence = population_prevalence
        self.pipeline_: Pipeline | None = None
        self.guard_ = ExtrapolationGuard(quantile=self.cfg.model.abstain_extrapolation_quantile)
        self.intercept_offset_: float = 0.0
        self.n_training_: int = 0
        self.sample_prevalence_: float = np.nan
        self.conformal_scores_: np.ndarray | None = None
        self.result_: DeclineModelResult | None = None

    # -- fitting ----------------------------------------------------------

    def _design(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        frame = data.copy()
        if "log_BT_ratio" in self.features and "log_BT_ratio" not in frame.columns:
            frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
        needed = [f for f in self.features if f in frame.columns]
        missing = set(self.features) - set(needed)
        if missing:
            raise ValueError(f"Missing required predictors: {sorted(missing)}")
        frame = frame.dropna(subset=needed + ["declined"])
        return frame[needed].astype(float), frame["declined"].astype(int)

    def _best_C(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
        """Pick the L2 strength by inner cross-validation, not by eye."""
        if len(np.unique(y)) < 2 or len(y) < 20:
            return 1.0
        best_C, best_score = 1.0, -np.inf
        splitter = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=int(rng.integers(1e6)))
        for C in self.cfg.model.logistic_C_grid:
            scores = []
            for train_idx, test_idx in splitter.split(X, y):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
                ])
                pipe.fit(X[train_idx], y[train_idx])
                probability = pipe.predict_proba(X[test_idx])[:, 1]
                if len(np.unique(y[test_idx])) < 2:
                    continue
                scores.append(roc_auc_score(y[test_idx], probability))
            if scores and np.mean(scores) > best_score:
                best_score, best_C = float(np.mean(scores)), C
        return best_C

    def fit(self, data: pd.DataFrame) -> "DeclineModel":
        X_frame, y_series = self._design(data)
        X, y = X_frame.to_numpy(float), y_series.to_numpy(int)
        rng = np.random.default_rng(self.cfg.model.random_seed)

        self.n_training_ = len(y)
        self.sample_prevalence_ = float(y.mean()) if len(y) else np.nan

        C = self._best_C(X, y, rng)
        self.pipeline_ = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
        ]).fit(X, y)
        self.guard_.fit(X_frame)

        # Case-control prior correction.
        if self.population_prevalence is not None:
            self.intercept_offset_ = prior_correction_offset(
                self.sample_prevalence_, self.population_prevalence
            )
        else:
            self.intercept_offset_ = 0.0

        # Conformal non-conformity scores: 1 - p(true class), used for the
        # distribution-free prediction set at the configured coverage.
        in_sample = self.pipeline_.predict_proba(X)[:, 1]
        self.conformal_scores_ = np.where(y == 1, 1 - in_sample, in_sample)

        self.result_ = self._evaluate(X_frame, X, y, C, rng)
        return self

    # -- evaluation -------------------------------------------------------

    def _cv_predictions(
        self, X: np.ndarray, y: np.ndarray, C: float, light: bool = False, seed: int | None = None
    ) -> tuple[np.ndarray, str]:
        """Out-of-fold predicted probabilities. LOO when n is very small.

        ``light=True`` uses a single stratified k-fold pass instead of repeated
        CV or leave-one-out. The permutation test calls it that way: the null
        distribution needs many cheap replicates, not one expensive estimate,
        and the observed AUC it is compared against is computed under the same
        scheme so the comparison stays like-for-like.
        """
        n = len(y)
        minority = int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 0
        oof = np.full(n, np.nan)

        if light:
            folds = max(2, min(5, minority))
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True,
                random_state=seed if seed is not None else self.cfg.model.random_seed,
            )
            for train_idx, test_idx in splitter.split(X, y):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
                ]).fit(X[train_idx], y[train_idx])
                oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
            return oof, f"{folds}-fold (light, permutation testing)"

        if n <= self.cfg.model.loo_threshold_n or minority < self.cfg.model.cv_folds:
            splitter, scheme = LeaveOneOut(), "leave-one-out"
            for train_idx, test_idx in splitter.split(X):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
                ]).fit(X[train_idx], y[train_idx])
                oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
            return oof, scheme

        folds = min(self.cfg.model.cv_folds, minority)
        scheme = f"{folds}-fold x {self.cfg.model.cv_repeats} repeats (stratified, participant-disjoint)"
        splitter = RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=self.cfg.model.cv_repeats,
            random_state=self.cfg.model.random_seed,
        )
        accumulated = np.zeros(n)
        counts = np.zeros(n)
        for train_idx, test_idx in splitter.split(X, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
            ]).fit(X[train_idx], y[train_idx])
            accumulated[test_idx] += pipe.predict_proba(X[test_idx])[:, 1]
            counts[test_idx] += 1
        oof = np.where(counts > 0, accumulated / np.maximum(counts, 1), np.nan)
        return oof, scheme

    def _bootstrap_coefficients(
        self, X: np.ndarray, y: np.ndarray, C: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Percentile bootstrap CIs and two-sided p-values for the coefficients."""
        n_boot = self.cfg.model.n_bootstrap
        draws = np.full((n_boot, X.shape[1]), np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
                ]).fit(X[idx], y[idx])
                draws[b] = pipe.named_steps["clf"].coef_[0]
            except Exception:
                continue
        valid = draws[~np.isnan(draws).any(axis=1)]
        if len(valid) < 50:
            shape = X.shape[1]
            return np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
        lower = np.percentile(valid, 2.5, axis=0)
        upper = np.percentile(valid, 97.5, axis=0)
        # Bootstrap p-value: proportion of draws on the far side of zero.
        p = 2 * np.minimum((valid <= 0).mean(axis=0), (valid >= 0).mean(axis=0))
        return lower, upper, np.clip(p, 1.0 / len(valid), 1.0)

    def _permutation_auc(self, X: np.ndarray, y: np.ndarray, C: float, n_perm: int = 500) -> tuple[float, float]:
        """How often does shuffled data reach this AUC? The honest null test.

        Both the observed and the permuted AUCs are computed under the same
        light CV scheme, so the p-value compares like with like. Returns
        ``(permutation_p, observed_auc_under_light_scheme)``.
        """
        rng = np.random.default_rng(self.cfg.model.random_seed + 7)

        observed_oof, _ = self._cv_predictions(X, y, C, light=True)
        mask = np.isfinite(observed_oof)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            return np.nan, np.nan
        observed = float(roc_auc_score(y[mask], observed_oof[mask]))

        hits = 0
        used = 0
        for i in range(n_perm):
            permuted = rng.permutation(y)
            oof, _ = self._cv_predictions(X, permuted, C, light=True, seed=int(rng.integers(1e6)))
            valid = np.isfinite(oof)
            if valid.sum() < 10 or len(np.unique(permuted[valid])) < 2:
                continue
            used += 1
            if roc_auc_score(permuted[valid], oof[valid]) >= observed:
                hits += 1
        if used == 0:
            return np.nan, observed
        return (hits + 1) / (used + 1), observed

    def _evaluate(
        self, X_frame: pd.DataFrame, X: np.ndarray, y: np.ndarray, C: float, rng: np.random.Generator
    ) -> DeclineModelResult:
        warnings: list[str] = []
        oof, scheme = self._cv_predictions(X, y, C)
        mask = np.isfinite(oof)

        if mask.sum() >= 10 and len(np.unique(y[mask])) > 1:
            auc = float(roc_auc_score(y[mask], oof[mask]))
            brier = float(brier_score_loss(y[mask], oof[mask]))
            # Fold-level SD via a lighter repeated split.
            fold_aucs = []
            splitter = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=1)
            for _, test_idx in splitter.split(X, y):
                sub = test_idx[np.isfinite(oof[test_idx])]
                if len(sub) > 5 and len(np.unique(y[sub])) > 1:
                    fold_aucs.append(roc_auc_score(y[sub], oof[sub]))
            auc_sd = float(np.std(fold_aucs)) if fold_aucs else np.nan
        else:
            auc, brier, auc_sd = np.nan, np.nan, np.nan
            warnings.append("Cross-validation produced too few usable predictions to score.")

        n_perm = 500
        perm_p, auc_light = self._permutation_auc(X, y, C, n_perm=n_perm)

        # Calibration in deciles of predicted risk.
        calibration = pd.DataFrame()
        if mask.sum() >= 20:
            frame = pd.DataFrame({"p": oof[mask], "y": y[mask]})
            n_bins = min(5, max(2, mask.sum() // 15))
            frame["bin"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop")
            calibration = (
                frame.groupby("bin")
                .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
                .reset_index(drop=True)
            )

        clf: LogisticRegression = self.pipeline_.named_steps["clf"]
        scale: StandardScaler = self.pipeline_.named_steps["scale"]
        # Coefficients are reported on the standardised scale (per 1 SD), which
        # is the only scale on which log(B/T), age and sex are comparable.
        coefs = clf.coef_[0]
        lower, upper, p_values = self._bootstrap_coefficients(X, y, C, rng)

        coefficients = pd.DataFrame(
            {
                "term": list(X_frame.columns),
                "coefficient": coefs,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_values,
                "odds_ratio": np.exp(coefs),
                "or_lower": np.exp(lower),
                "or_upper": np.exp(upper),
                "scale": "per 1 SD of predictor",
            }
        )

        if np.isfinite(perm_p) and perm_p >= 0.05:
            verdict = (
                "This model does not predict grip decline better than chance. "
                f"Cross-validated AUC was {auc:.2f}, but shuffled data reached that "
                f"level {perm_p * 100:.0f}% of the time (permutation p = {perm_p:.3f}). "
                "No individual risk estimate from this model should be acted on."
            )
        elif np.isfinite(perm_p):
            verdict = (
                f"The model separates decliners from stable participants better than chance "
                f"(AUC {auc:.2f}, permutation p = {perm_p:.3f}). "
                "This is a within-sample case-control result and needs external replication "
                "before it means anything clinically."
            )
        else:
            verdict = "The model could not be validated - too few usable observations."

        technical = (
            f"L2 logistic regression (C = {C}), n = {len(y)} "
            f"({int(y.sum())} decliners / {int((1 - y).sum())} stable). "
            f"Validation: {scheme}. AUC = {auc:.3f}"
            + (f" (SD {auc_sd:.3f})" if np.isfinite(auc_sd) else "")
            + f", Brier = {brier:.3f}. Permutation test: observed AUC {auc_light:.3f} under a "
            f"matched 5-fold scheme, p = {perm_p:.3f} over {n_perm} label permutations."
        )

        if self.population_prevalence is not None:
            warnings.append(
                f"Sampling was case-control ({self.sample_prevalence_ * 100:.0f}% decliners by "
                f"design). Predicted probabilities have been prior-corrected to the observed "
                f"cohort prevalence of {self.population_prevalence * 100:.1f}%."
            )
        else:
            warnings.append(
                "Sampling was case-control. Predicted probabilities reflect the 50/50 study "
                "sample and are NOT population risks. No prevalence was supplied to correct them."
            )

        return DeclineModelResult(
            coefficients=coefficients,
            auc_cv=auc,
            auc_cv_sd=auc_sd,
            auc_permutation_p=perm_p,
            brier=brier,
            calibration=calibration,
            n=len(y),
            n_cases=int(y.sum()),
            sample_prevalence=self.sample_prevalence_,
            population_prevalence=self.population_prevalence,
            intercept_offset=self.intercept_offset_,
            cv_scheme=scheme,
            verdict=verdict,
            technical_note=technical,
            warnings=warnings,
        )

    # -- prediction -------------------------------------------------------

    def predict_risk(self, profile: dict | pd.Series, apply_correction: bool = True) -> Prediction:
        """Individual decline risk, or a refusal with reasons."""
        prediction = Prediction(units="%", coverage=1 - self.cfg.model.conformal_alpha, interval_kind="prediction")
        if self.pipeline_ is None or self.result_ is None:
            return prediction.abstain("The model has not been fitted.")

        check_training_adequacy(self.n_training_, self.cfg.model.abstain_min_training_n, prediction)

        # A model that fails its own permutation test must not issue risks.
        if self.cfg.strict_evidence_mode and not self.result_.is_predictive:
            prediction.abstain(
                "This model does not beat chance on validation "
                f"(permutation p = {self.result_.auc_permutation_p:.3f}). "
                "Issuing an individual risk from it would be fabricating precision."
            )

        if isinstance(profile, dict) and "BT_ratio" in profile and "log_BT_ratio" not in profile:
            ratio = profile.get("BT_ratio")
            if ratio is not None and ratio > 0:
                profile = {**profile, "log_BT_ratio": float(np.log(ratio))}

        inside, messages = self.guard_.check(profile)
        if not inside:
            for message in messages:
                prediction.abstain(message)

        if prediction.abstained:
            prediction.provenance = {
                "model": "L2 logistic regression",
                "n_training": self.n_training_,
                "auc_cv": self.result_.auc_cv,
                "permutation_p": self.result_.auc_permutation_p,
            }
            return prediction

        values = np.array([[float(profile[name]) for name in self.guard_.feature_names_]])
        logit = float(self.pipeline_.decision_function(values)[0])
        if apply_correction and self.intercept_offset_:
            logit -= self.intercept_offset_
            prediction.caveat(
                "Risk has been prior-corrected from the 50/50 case-control design to the "
                f"observed cohort prevalence ({self.population_prevalence * 100:.1f}%)."
            )
        else:
            prediction.caveat(
                "This probability reflects the 50/50 case-control sample, NOT population risk."
            )

        probability = 1 / (1 + np.exp(-logit))
        prediction.value = float(probability * 100)

        # Conformal interval from the calibration scores.
        if self.conformal_scores_ is not None and len(self.conformal_scores_) >= 20:
            alpha = self.cfg.model.conformal_alpha
            quantile = float(np.quantile(self.conformal_scores_, 1 - alpha))
            prediction.lower = float(max(0.0, (probability - quantile)) * 100)
            prediction.upper = float(min(1.0, (probability + quantile)) * 100)
            if (prediction.upper - prediction.lower) > 70:
                prediction.abstain(
                    "The uncertainty around this estimate spans more than 70 percentage "
                    "points. There is no usable prediction here."
                )

        prediction.provenance = {
            "model": "L2 logistic regression (primary)",
            "features": self.guard_.feature_names_,
            "n_training": self.n_training_,
            "auc_cv": self.result_.auc_cv,
            "permutation_p": self.result_.auc_permutation_p,
            "prior_corrected": bool(apply_correction and self.intercept_offset_),
        }
        return prediction


# --------------------------------------------------------------------------
# Sensitivity model (never the headline)
# --------------------------------------------------------------------------

def gradient_boosting_sensitivity(
    data: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Exploratory gradient-boosting check with SHAP importances.

    Labelled EXPLORATORY throughout. With n=100 this model cannot be the
    primary result and the returned dict says so explicitly.
    """
    cfg = settings or SETTINGS
    output: dict[str, Any] = {
        "label": "EXPLORATORY ROBUSTNESS CHECK - NOT THE PRIMARY RESULT",
        "available": False,
    }
    if not cfg.model.enable_gradient_boosting:
        output["reason"] = "Disabled in configuration."
        return output

    try:
        import lightgbm as lgb
    except ImportError:
        output["reason"] = "LightGBM is not installed."
        return output

    frame = data.copy()
    if "log_BT_ratio" not in frame.columns and "BT_ratio" in frame.columns:
        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
    usable = [f for f in features if f in frame.columns]
    frame = frame.dropna(subset=usable + ["declined"])
    if len(frame) < 40:
        output["reason"] = f"Only {len(frame)} complete cases - too few for a tree ensemble."
        return output

    X = frame[usable].to_numpy(float)
    y = frame["declined"].to_numpy(int)

    params = dict(
        n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=3,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, verbose=-1, random_state=cfg.model.random_seed,
    )

    folds = min(5, int(np.bincount(y).min()))
    oof = np.full(len(y), np.nan)
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.model.random_seed)
        for train_idx, test_idx in splitter.split(X, y):
            model = lgb.LGBMClassifier(**params).fit(X[train_idx], y[train_idx])
            oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    mask = np.isfinite(oof)
    auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() > 10 and len(np.unique(y[mask])) > 1 else np.nan

    final = lgb.LGBMClassifier(**params).fit(X, y)
    output.update({"available": True, "auc_cv": auc, "n": len(y), "features": usable})

    if cfg.model.enable_shap:
        try:
            import shap

            explainer = shap.TreeExplainer(final)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            importance = np.abs(shap_values).mean(axis=0)
            output["shap_importance"] = pd.DataFrame(
                {"feature": usable, "mean_abs_shap": importance}
            ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        except Exception as exc:  # pragma: no cover
            output["shap_error"] = str(exc)

    output["interpretation"] = (
        f"Gradient boosting reached a cross-validated AUC of {auc:.2f}. "
        "With this sample size that figure is unstable and is reported only to show "
        "whether a flexible model finds structure the logistic regression missed. "
        "It does not replace the primary model."
    )
    return output


def fit_decline_model(
    substudy: pd.DataFrame,
    wide: pd.DataFrame | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    settings: Settings | None = None,
    run_sensitivity: bool = True,
) -> DeclineModel:
    """Fit the primary decline model, deriving cohort prevalence from ``wide``.

    The population prevalence used for the prior correction comes from the full
    grip cohort, never from the case-control substudy - which is 50% by
    construction and would correct nothing.
    """
    cfg = settings or SETTINGS

    population_prevalence = None
    if wide is not None and "declined" in wide.columns:
        observed = wide["declined"].dropna()
        if len(observed) >= 50:
            population_prevalence = float(observed.mean())

    model = DeclineModel(features=features, settings=cfg, population_prevalence=population_prevalence)
    model.fit(substudy)

    if run_sensitivity and model.result_ is not None:
        model.result_.sensitivity = gradient_boosting_sensitivity(substudy, features, cfg)

    return model
