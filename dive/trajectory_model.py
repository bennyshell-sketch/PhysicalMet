"""Longitudinal trajectory model for physical function.

Linear mixed-effects model fitted on the fuller longitudinal cohort - the
participants with repeated grip measurements, not the n=100 case-control
substudy. Case-control sampling distorts prevalence and the slope
distribution; the trajectory question needs the real cohort.

    function ~ time * (log B/T + age + sex) + (1 + time | participant)

Time is **actual elapsed months**, never a visit index. PORAF visits are
irregular by design, and treating a 4.5-month "6-month" visit as 6.0 months
pushes the estimated slope toward zero.

Forecasts are returned as **prediction intervals**, not confidence intervals.
A confidence interval covers where the group mean sits; a prediction interval
covers where *this person* is likely to land, which is what a forecast is.
The two differ by the random-effect and residual variance, and for individual
trajectories that difference is most of the width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .guardrails import ExtrapolationGuard, Prediction

__all__ = ["TrajectoryModel", "TrajectoryResult", "fit_trajectory_model"]

#: Visits are recorded in months; the model works in years (see ``_prepare``).
MONTHS_PER_YEAR = 12.0

#: Tried in order until one converges. lbfgs is fastest but is the one that
#: fails on poorly scaled variance components, so powell backs it up.
_OPTIMISERS = ("lbfgs", "powell", "cg", "bfgs")


def _fit_with_fallback(model, maxiter: int = 3000):
    """Fit a MixedLM, trying successive optimisers until one converges.

    Returns ``(fit, converged)``. If none converge, the last fit is returned
    with ``converged=False`` so the caller can flag the estimates as
    provisional rather than silently presenting them as final.
    """
    import warnings as _warnings

    last = None
    for method in _OPTIMISERS:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                fit = model.fit(method=method, maxiter=maxiter, reml=True)
        except Exception:
            continue
        last = fit
        if bool(getattr(fit, "converged", False)):
            return fit, True
    if last is None:
        raise RuntimeError("MixedLM failed to fit under every optimiser.")
    return last, False


@dataclass
class TrajectoryResult:
    fixed_effects: pd.DataFrame
    random_effect_sd: dict[str, float]
    residual_sd: float
    n_participants: int
    n_observations: int
    converged: bool
    outcome: str
    verdict: str
    technical_note: str
    marginal_r2: float = np.nan
    conditional_r2: float = np.nan
    warnings: list[str] = field(default_factory=list)

    def fixed_effects_table(self) -> pd.DataFrame:
        table = self.fixed_effects.copy()
        table["Estimate (95% CI)"] = [
            f"{row['estimate']:.4f} ({row['ci_lower']:.4f} to {row['ci_upper']:.4f})"
            for _, row in table.iterrows()
        ]
        table["p"] = table["p_value"].map(
            lambda v: "n/a" if not np.isfinite(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")
        )
        return table[["term", "Estimate (95% CI)", "p"]].rename(columns={"term": "Term"})


class TrajectoryModel:
    """Mixed-effects trajectory model with individual prediction intervals."""

    def __init__(
        self,
        outcome: str = "max_grip_kg",
        predictors: Sequence[str] = ("log_BT_ratio", "age_years", "sex"),
        settings: Settings | None = None,
    ):
        self.cfg = settings or SETTINGS
        self.outcome = outcome
        self.predictors = list(predictors)
        self.fit_: Any = None
        self.model_: Any = None
        self.design_columns_: list[str] = []
        self.used_predictors_: list[str] = []
        self.guard_ = ExtrapolationGuard(quantile=self.cfg.model.abstain_extrapolation_quantile)
        self.result_: TrajectoryResult | None = None
        self.training_frame_: pd.DataFrame | None = None
        self.re_cov_: np.ndarray | None = None
        self.residual_var_: float = np.nan
        self.centres_: dict[str, float] = {}

    # -- fitting ----------------------------------------------------------

    def _prepare(self, long: pd.DataFrame) -> pd.DataFrame:
        frame = long.copy()
        if "log_BT_ratio" in self.predictors and "log_BT_ratio" not in frame.columns:
            if "BT_ratio" in frame.columns:
                frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))

        # Only keep predictors that actually exist AND vary. In the default
        # pipeline the B/T ratio is measured on a subset, so including it would
        # silently discard every participant without an ELISA result. That
        # decision is made explicitly and reported, not hidden in a dropna.
        usable: list[str] = []
        for name in self.predictors:
            if name not in frame.columns:
                continue
            coverage = frame[name].notna().mean()
            if coverage < 0.5:
                continue
            usable.append(name)
        self.used_predictors_ = usable

        needed = [self.outcome, "elapsed_months", "participant_id"] + usable
        frame = frame.dropna(subset=[c for c in needed if c in frame.columns]).copy()

        # Centre continuous predictors so the intercept means "an average
        # participant at baseline", not "a zero-year-old".
        # Every predictor is centred, sex included. With sex left at 0/1 the
        # `time` coefficient would be the slope for males only; centring makes
        # it the population-average slope, which is what the verdict claims it
        # is. The sex coefficient still reads as the male-female difference.
        self.centres_ = {}
        for name in usable:
            centre = float(frame[name].mean())
            self.centres_[name] = centre
            frame[f"c_{name}"] = frame[name] - centre

        # Time is carried in YEARS internally even though visits are recorded in
        # months. With months, the random-slope variance is ~10,000x smaller
        # than the random-intercept variance and the REML optimiser fails to
        # identify it - it returns slope SDs orders of magnitude too large.
        # Rescaling puts the two variance components within ~100x of each
        # other, and the fit becomes stable across optimisers. Fixed-effect
        # slopes are therefore in units per YEAR.
        frame["time"] = frame["elapsed_months"].astype(float) / MONTHS_PER_YEAR
        return frame

    def fit(self, long: pd.DataFrame) -> "TrajectoryModel":
        import statsmodels.formula.api as smf

        frame = self._prepare(long)
        repeated = frame.groupby("participant_id")[self.outcome].transform("size")
        frame = frame[repeated >= 2].copy()
        self.training_frame_ = frame

        warnings_out: list[str] = []
        dropped = [p for p in self.predictors if p not in self.used_predictors_]
        if dropped:
            warnings_out.append(
                f"Predictors excluded for insufficient coverage: {', '.join(dropped)}. "
                "Including them would have restricted the model to the assayed subsample "
                "and thrown away most of the longitudinal cohort."
            )

        terms = ["time"]
        for name in self.used_predictors_:
            column = f"c_{name}"
            terms.append(column)
            terms.append(f"time:{column}")
        formula = f"Q('{self.outcome}') ~ " + " + ".join(terms)

        model = smf.mixedlm(
            formula, frame, groups=frame["participant_id"], re_formula="1 + time"
        )
        try:
            fit, converged = _fit_with_fallback(model)
        except Exception:
            # Random slopes need enough within-participant timepoints. If the
            # covariance is not identifiable, fall back to a random intercept
            # and say so, rather than reporting an unstable slope variance.
            model = smf.mixedlm(formula, frame, groups=frame["participant_id"])
            fit, converged = _fit_with_fallback(model)
            warnings_out.append(
                "Random-slope covariance was not identifiable; fell back to a "
                "random-intercept model. Individual slope forecasts are therefore "
                "shrunk toward the cohort mean slope."
            )

        self.model_, self.fit_ = model, fit
        if not converged:
            warnings_out.append("The mixed model did not fully converge; treat estimates as provisional.")

        # statsmodels reports cov_re already in the outcome's own units;
        # rescaling it by the residual variance would inflate every interval.
        self.re_cov_ = np.atleast_2d(np.asarray(fit.cov_re))
        self.residual_var_ = float(fit.scale)

        design_predictors = frame[[c for c in frame.columns if c.startswith("c_")]]
        if not design_predictors.empty:
            self.guard_.fit(design_predictors.groupby(frame["participant_id"]).first())

        self.result_ = self._summarise(fit, frame, converged, warnings_out)
        return self

    def _summarise(self, fit, frame, converged, warnings_out) -> TrajectoryResult:
        params = fit.params
        conf = fit.conf_int()
        pvalues = fit.pvalues

        rows = []
        for term in params.index:
            if term.endswith("Var") or "Cov" in term:
                continue
            label = term.replace("c_", "").replace("Q('", "").replace("')", "")
            rows.append(
                {
                    "term": label,
                    "estimate": float(params[term]),
                    "ci_lower": float(conf.loc[term, 0]),
                    "ci_upper": float(conf.loc[term, 1]),
                    "p_value": float(pvalues.get(term, np.nan)),
                }
            )
        fixed = pd.DataFrame(rows)

        re_sd = {}
        cov = np.atleast_2d(np.asarray(fit.cov_re))
        names = list(fit.cov_re.index) if hasattr(fit.cov_re, "index") else ["Intercept"]
        for i, name in enumerate(names[: cov.shape[0]]):
            re_sd[str(name)] = float(np.sqrt(max(cov[i, i], 0.0)))
        residual_sd = float(np.sqrt(fit.scale))

        # Nakagawa marginal / conditional R^2.
        try:
            fitted_fixed = np.asarray(fit.fittedvalues) - np.asarray(fit.resid) * 0
            var_fixed = float(np.var(fit.model.exog @ fit.fe_params))
            var_re = float(cov[0, 0])
            total = var_fixed + var_re + fit.scale
            marginal_r2 = var_fixed / total
            conditional_r2 = (var_fixed + var_re) / total
        except Exception:
            marginal_r2 = conditional_r2 = np.nan

        slope_row = fixed[fixed["term"] == "time"]
        slope = float(slope_row["estimate"].iloc[0]) if not slope_row.empty else np.nan

        bt_terms = [r for _, r in fixed.iterrows() if "BT" in r["term"] and "time" in r["term"]]
        if bt_terms:
            bt = bt_terms[0]
            if np.isfinite(bt["p_value"]) and bt["p_value"] < 0.05:
                bt_sentence = (
                    f" The B/T ratio is associated with the rate of change "
                    f"({bt['estimate']:+.4f} kg/year per unit log B/T, p = {bt['p_value']:.3f})."
                )
            else:
                bt_sentence = (
                    " The B/T ratio shows no association with the rate of change "
                    f"(p = {bt['p_value']:.3f}); trajectories look the same regardless of it."
                )
        else:
            bt_sentence = " The B/T ratio was not included in this model (see warnings)."

        direction = "losing" if slope < 0 else "gaining"
        verdict = (
            f"On average the cohort is {direction} "
            f"{abs(slope):.2f} kg of grip strength per year."
            + bt_sentence
        )

        technical = (
            f"Linear mixed-effects model, {frame['participant_id'].nunique()} participants, "
            f"{len(frame)} observations. Random intercept SD "
            f"{re_sd.get('Group', re_sd.get('Intercept', float('nan'))):.3f}, "
            f"residual SD {residual_sd:.3f}. "
            f"Marginal R2 = {marginal_r2:.3f}, conditional R2 = {conditional_r2:.3f}. "
            "Time is actual elapsed months since baseline."
        )

        return TrajectoryResult(
            fixed_effects=fixed,
            random_effect_sd=re_sd,
            residual_sd=residual_sd,
            n_participants=int(frame["participant_id"].nunique()),
            n_observations=int(len(frame)),
            converged=converged,
            outcome=self.outcome,
            verdict=verdict,
            technical_note=technical,
            marginal_r2=marginal_r2,
            conditional_r2=conditional_r2,
            warnings=warnings_out,
        )

    # -- prediction -------------------------------------------------------

    def _design_row(self, profile: dict, months: float) -> tuple[np.ndarray, list[str]]:
        """Build one design-matrix row matching the fitted model's terms."""
        names = list(self.fit_.fe_params.index)
        row = np.zeros(len(names))
        problems: list[str] = []

        for i, term in enumerate(names):
            if term == "Intercept":
                row[i] = 1.0
            elif term == "time":
                row[i] = months / MONTHS_PER_YEAR
            elif ":" in term:
                left, right = term.split(":")
                years = months / MONTHS_PER_YEAR
                left_value = years if left == "time" else self._value_for(left, profile, problems)
                right_value = years if right == "time" else self._value_for(right, profile, problems)
                row[i] = left_value * right_value
            else:
                row[i] = self._value_for(term, profile, problems)
        return row, problems

    def _value_for(self, term: str, profile: dict, problems: list[str]) -> float:
        raw_name = term[2:] if term.startswith("c_") else term
        if raw_name not in profile or profile[raw_name] is None:
            problems.append(f"{raw_name} is required for this forecast but was not supplied.")
            return 0.0
        value = float(profile[raw_name])
        if term.startswith("c_"):
            value -= self.centres_.get(raw_name, 0.0)
        return value

    def predict_trajectory(
        self,
        profile: dict,
        months: Sequence[float] | None = None,
        observed: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast a trajectory with 95% prediction intervals.

        ``observed`` (columns ``elapsed_months`` and the outcome) makes the
        forecast participant-specific: the person's own random intercept and
        slope are estimated by empirical-Bayes shrinkage and the interval
        narrows accordingly. Without it the forecast is for an otherwise
        unremarkable person with this covariate profile, and the interval
        carries the full between-participant variance.
        """
        if self.fit_ is None:
            raise RuntimeError("Model has not been fitted.")

        horizon = months if months is not None else np.arange(
            0, self.cfg.model.trajectory_horizon_months + 0.5, 0.5
        )
        horizon = np.asarray(horizon, dtype=float)

        z = 1.959963985
        cov_fe = np.asarray(self.fit_.cov_params())
        n_fe = len(self.fit_.fe_params)
        cov_fe = cov_fe[:n_fe, :n_fe]

        re_cov = np.atleast_2d(self.re_cov_)
        if re_cov.shape[0] == 1:
            re_cov = np.array([[re_cov[0, 0], 0.0], [0.0, 0.0]])

        # Empirical-Bayes BLUP for this participant, if history was supplied.
        blup = np.zeros(2)
        blup_cov = re_cov.copy()
        used_history = False
        if observed is not None and len(observed.dropna()) >= 2:
            history = observed.dropna().copy()
            t = history["elapsed_months"].to_numpy(float)
            y = history[self.outcome].to_numpy(float)
            fixed_at_t = np.array([self._design_row(profile, tt)[0] for tt in t])
            residual = y - fixed_at_t @ np.asarray(self.fit_.fe_params)
            Z = np.column_stack([np.ones_like(t), t / MONTHS_PER_YEAR])
            V = Z @ re_cov @ Z.T + np.eye(len(t)) * self.residual_var_
            try:
                V_inv = np.linalg.inv(V)
                blup = re_cov @ Z.T @ V_inv @ residual
                blup_cov = re_cov - re_cov @ Z.T @ V_inv @ Z @ re_cov
                used_history = True
            except np.linalg.LinAlgError:
                pass

        records = []
        for month in horizon:
            row, problems = self._design_row(profile, month)
            if problems:
                raise ValueError("; ".join(sorted(set(problems))))
            fixed_part = float(row @ np.asarray(self.fit_.fe_params))
            z_row = np.array([1.0, month / MONTHS_PER_YEAR])
            point = fixed_part + float(z_row @ blup)

            var_fixed = float(row @ cov_fe @ row)
            var_re = float(z_row @ blup_cov @ z_row)
            # Confidence interval: uncertainty in the mean only.
            var_conf = var_fixed + (var_re if used_history else 0.0)
            # Prediction interval: adds between-participant and residual variance.
            var_pred = var_fixed + var_re + self.residual_var_

            records.append(
                {
                    "months": month,
                    "predicted": point,
                    "ci_lower": point - z * np.sqrt(max(var_conf, 0.0)),
                    "ci_upper": point + z * np.sqrt(max(var_conf, 0.0)),
                    "pi_lower": point - z * np.sqrt(max(var_pred, 0.0)),
                    "pi_upper": point + z * np.sqrt(max(var_pred, 0.0)),
                }
            )

        frame = pd.DataFrame.from_records(records)
        frame.attrs["participant_specific"] = used_history
        frame.attrs["outcome"] = self.outcome
        return frame

    def forecast(
        self,
        profile: dict,
        months: float | None = None,
        observed: pd.DataFrame | None = None,
    ) -> Prediction:
        """Single-horizon forecast wrapped in the abstention envelope."""
        horizon = months if months is not None else self.cfg.model.trajectory_horizon_months
        prediction = Prediction(
            units="kg" if "grip" in self.outcome else "",
            coverage=0.95,
            interval_kind="prediction",
        )

        if self.fit_ is None or self.result_ is None:
            return prediction.abstain("The trajectory model has not been fitted.")

        if horizon > self.cfg.model.trajectory_max_horizon_months:
            return prediction.abstain(
                f"A {horizon:.0f}-month forecast is beyond the "
                f"{self.cfg.model.trajectory_max_horizon_months:.0f}-month limit supported "
                "by the observed follow-up. Extrapolating further would be invention."
            )

        guard_profile = {}
        for name in self.guard_.feature_names_:
            raw = name[2:] if name.startswith("c_") else name
            if raw in profile and profile[raw] is not None:
                value = float(profile[raw])
                guard_profile[name] = value - self.centres_.get(raw, 0.0) if name.startswith("c_") else value
        inside, messages = self.guard_.check(guard_profile) if guard_profile else (True, [])
        if not inside:
            for message in messages:
                prediction.abstain(message)
            return prediction

        try:
            frame = self.predict_trajectory(profile, months=[horizon], observed=observed)
        except ValueError as exc:
            return prediction.abstain(str(exc))

        row = frame.iloc[0]
        prediction.value = float(row["predicted"])
        prediction.lower = float(row["pi_lower"])
        prediction.upper = float(row["pi_upper"])

        width = prediction.upper - prediction.lower
        limit = self.cfg.model.abstain_interval_sd_multiple * 2 * self.result_.residual_sd
        if width > limit * 3:
            prediction.abstain(
                f"The 95% prediction interval spans {width:.1f} {prediction.units}, which is "
                "too wide to inform any decision."
            )

        if frame.attrs.get("participant_specific"):
            prediction.caveat(
                "Forecast is personalised using this participant's own measurement history."
            )
        else:
            prediction.caveat(
                "No measurement history was supplied, so this is the expected trajectory for "
                "someone with these characteristics, not for a specific individual. The "
                "interval is correspondingly wide."
            )

        prediction.provenance = {
            "model": "Linear mixed-effects (random intercept + slope)",
            "n_participants": self.result_.n_participants,
            "n_observations": self.result_.n_observations,
            "predictors": self.used_predictors_,
            "horizon_months": horizon,
            "participant_specific": bool(frame.attrs.get("participant_specific")),
        }
        return prediction


def fit_trajectory_model(
    long: pd.DataFrame,
    outcome: str = "max_grip_kg",
    predictors: Sequence[str] = ("log_BT_ratio", "age_years", "sex"),
    settings: Settings | None = None,
) -> TrajectoryModel:
    """Fit the trajectory model on the longitudinal cohort."""
    return TrajectoryModel(outcome=outcome, predictors=predictors, settings=settings).fit(long)
