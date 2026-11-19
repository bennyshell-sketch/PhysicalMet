"""Competence checks that stop the system inventing predictions.

The rule this module enforces: **a prediction is only released if the model
has grounds to make it.** Everything else is refused, with the reason stated.

Three independent checks run before any individual prediction is returned:

1. *Training adequacy* - was the model fitted on enough data at all?
2. *Extrapolation* - does this person's covariate profile sit inside the
   region the model actually saw? Both per-feature range and multivariate
   (Mahalanobis) distance are checked, because a person can be individually
   unremarkable on every variable yet be a combination nobody in the sample
   resembled.
3. *Interval width* - is the honest uncertainty so wide that the number
   carries no information? A "predicted grip of 12-46 kg" is not a prediction.

A refusal is a legitimate output, not an error. It is more useful to a
clinician than a confident number with no support behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = ["Prediction", "ExtrapolationGuard", "check_training_adequacy"]


# --------------------------------------------------------------------------
# Prediction envelope
# --------------------------------------------------------------------------

@dataclass
class Prediction:
    """A prediction, its uncertainty, and the grounds on which it was issued.

    ``abstained=True`` means no number is being asserted. ``value`` may still
    be populated for diagnostics, but callers must not display it as a result
    when ``abstained`` is set.
    """

    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    interval_kind: str = "prediction"      # "prediction" | "confidence"
    coverage: float = 0.95
    units: str = ""
    abstained: bool = False
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def abstain(self, reason: str) -> "Prediction":
        self.abstained = True
        self.reasons.append(reason)
        return self

    def caveat(self, note: str) -> "Prediction":
        if note not in self.caveats:
            self.caveats.append(note)
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def format_value(self, decimals: int = 1) -> str:
        if self.abstained or self.value is None:
            return "Not predicted"
        return f"{self.value:.{decimals}f} {self.units}".strip()

    def format_interval(self, decimals: int = 1) -> str:
        if self.abstained or self.lower is None or self.upper is None:
            return "-"
        return f"{self.lower:.{decimals}f} to {self.upper:.{decimals}f} {self.units}".strip()

    def headline(self, decimals: int = 1) -> str:
        """One line suitable for showing to a non-technical reader."""
        if self.abstained:
            return "No prediction issued - " + (self.reasons[0] if self.reasons else "outside model competence.")
        return (
            f"{self.format_value(decimals)} "
            f"({int(self.coverage * 100)}% {self.interval_kind} interval "
            f"{self.format_interval(decimals)})"
        )


# --------------------------------------------------------------------------
# Training adequacy
# --------------------------------------------------------------------------

def check_training_adequacy(n_training: int, minimum: int, prediction: Prediction) -> Prediction:
    """Refuse to predict from a model fitted on too little data."""
    if n_training < minimum:
        prediction.abstain(
            f"The model was fitted on only {n_training} participants "
            f"(minimum {minimum}). That is too few to support an individual prediction."
        )
    return prediction


# --------------------------------------------------------------------------
# Extrapolation
# --------------------------------------------------------------------------

class ExtrapolationGuard:
    """Detects covariate profiles the model was never trained on.

    Fitted on the training design matrix. ``check`` reports both per-feature
    range violations and a multivariate Mahalanobis distance compared against
    the training distribution's own quantile.
    """

    def __init__(self, quantile: float = 0.99, range_tolerance: float = 0.10):
        self.quantile = quantile
        self.range_tolerance = range_tolerance
        self.feature_names_: list[str] = []
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.threshold_: float = np.inf
        self.n_training_: int = 0

    def fit(self, X: pd.DataFrame | np.ndarray, feature_names: Sequence[str] | None = None) -> "ExtrapolationGuard":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            values = X.to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            self.feature_names_ = list(feature_names) if feature_names is not None else [
                f"x{i}" for i in range(values.shape[1])
            ]

        values = values[np.isfinite(values).all(axis=1)]
        self.n_training_ = len(values)
        if self.n_training_ == 0:
            return self

        # Observed range, widened by a tolerance so a value a hair outside the
        # training minimum is not treated as exotic.
        span = values.max(axis=0) - values.min(axis=0)
        pad = np.where(span > 0, span * self.range_tolerance, 1.0)
        self.lower_ = values.min(axis=0) - pad
        self.upper_ = values.max(axis=0) + pad

        self.mean_ = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        # Ridge the covariance so near-collinear features stay invertible.
        ridge = np.eye(covariance.shape[0]) * (np.trace(covariance) / covariance.shape[0]) * 1e-6
        try:
            self.precision_ = np.linalg.inv(covariance + ridge)
        except np.linalg.LinAlgError:
            self.precision_ = np.linalg.pinv(covariance + ridge)

        distances = self._mahalanobis(values)
        self.threshold_ = float(np.quantile(distances, self.quantile))
        return self

    def _mahalanobis(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            return np.zeros(len(values))
        centred = values - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", centred, self.precision_, centred))

    def check(self, x: pd.Series | np.ndarray | dict) -> tuple[bool, list[str]]:
        """Return ``(is_inside, messages)`` for one covariate profile."""
        if self.lower_ is None:
            return True, []

        if isinstance(x, dict):
            values = np.array([float(x.get(name, np.nan)) for name in self.feature_names_])
        elif isinstance(x, pd.Series):
            values = np.array([float(x.get(name, np.nan)) for name in self.feature_names_])
        else:
            values = np.asarray(x, dtype=float).ravel()

        messages: list[str] = []
        if not np.all(np.isfinite(values)):
            missing = [n for n, v in zip(self.feature_names_, values) if not np.isfinite(v)]
            return False, [f"Required input missing: {', '.join(missing)}."]

        below = values < self.lower_
        above = values > self.upper_
        for i, name in enumerate(self.feature_names_):
            if below[i] or above[i]:
                messages.append(
                    f"{name} = {values[i]:.3g} lies outside the range seen in training "
                    f"({self.lower_[i]:.3g} to {self.upper_[i]:.3g})."
                )

        distance = float(self._mahalanobis(values.reshape(1, -1))[0])
        if distance > self.threshold_:
            messages.append(
                "This combination of characteristics is unlike anyone in the training "
                f"sample (distance {distance:.2f}, limit {self.threshold_:.2f})."
            )

        return len(messages) == 0, messages
