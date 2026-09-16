"""Deciding whether a new health metric actually improves prediction.

This is the module that lets Dive get better as data arrives, without letting
it talk itself into improvements that are not there.

The naive version of this feature is a trap. Hand a system twenty new
variables, let it try each one, keep whichever raises cross-validated AUC the
most, and it will report an improvement essentially every time - because the
maximum of twenty noisy estimates is biased upward even when every variable is
pure noise. The more metrics you add, the more confident and the more wrong the
system becomes.

Three defences, applied together:

1. **Participant-disjoint folds.** A participant's repeated visits never span
   train and test.

2. **A permutation test on the whole selection procedure, not on one feature.**
   The outcome labels are shuffled and the *entire* search is re-run, including
   the "pick the best candidate" step. That null distribution answers the
   question that actually matters: how large an improvement would this search
   have found if none of these metrics carried any signal? A candidate is
   adopted only if its improvement beats that.

3. **A minimum effect size and a coverage floor.** A metric measured on 15% of
   participants cannot be evaluated honestly, and a +0.004 AUC gain is not a
   finding.

4. **A screen on protected attributes.** Self-reported ethnicity and race are
   refused as candidate features before anything is fitted. They are not
   biological variables, and the algorithms that treated them as such had to be
   withdrawn - see ``dive/fairness.py``. Genetic ancestry components are a
   different variable and are ordinary candidates.

The expensive part is defence 2 - it costs ``n_permutations`` x ``n_candidates``
model fits. That cost is the point. It is what separates a metric that helps
from one that happened to look good once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import SETTINGS, Settings
from ..validation import participant_aware_folds

__all__ = ["FeatureCandidate", "FeatureSearchResult", "evaluate_feature_set", "search_features"]


@dataclass
class FeatureCandidate:
    name: str
    coverage: float
    n_usable: int
    auc: float = float("nan")
    delta_auc: float = float("nan")
    evaluated: bool = False
    skip_reason: str = ""


@dataclass
class FeatureSearchResult:
    base_features: list[str]
    adopted_features: list[str]
    base_auc: float
    best_auc: float
    best_candidate: str | None
    delta_auc: float
    selection_p_value: float
    candidates: list[FeatureCandidate] = field(default_factory=list)
    verdict: str = ""
    n_permutations: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.best_candidate is not None and self.adopted_features != self.base_features

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for candidate in sorted(
            self.candidates, key=lambda c: (-(c.delta_auc if np.isfinite(c.delta_auc) else -9))
        ):
            rows.append(
                {
                    "Metric": candidate.name,
                    "Coverage": f"{candidate.coverage * 100:.0f}%",
                    "n": candidate.n_usable,
                    "AUC with metric": (
                        f"{candidate.auc:.3f}" if np.isfinite(candidate.auc) else "-"
                    ),
                    "Change": (
                        f"{candidate.delta_auc:+.3f}" if np.isfinite(candidate.delta_auc) else "-"
                    ),
                    "Note": candidate.skip_reason or "",
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _fit_score(
    frame: pd.DataFrame, features: Sequence[str], outcome: str, seed: int, n_splits: int
) -> float:
    """Out-of-fold AUC for one feature set, on participant-disjoint folds."""
    usable = frame.dropna(subset=list(features) + [outcome])
    if len(usable) < 30 or usable[outcome].nunique() < 2:
        return float("nan")

    X = usable[list(features)].to_numpy(float)
    y = usable[outcome].to_numpy(int)
    ids = usable["participant_id"].to_numpy()

    folds = participant_aware_folds(ids, n_splits=n_splits, seed=seed)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, max_iter=2000, solver="lbfgs")),
        ]).fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    mask = np.isfinite(oof)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], oof[mask]))


def evaluate_feature_set(
    data: pd.DataFrame,
    features: Sequence[str],
    outcome: str = "declined",
    settings: Settings | None = None,
    n_splits: int = 5,
) -> float:
    """Cross-validated AUC for a feature set. Public entry point."""
    cfg = settings or SETTINGS
    return _fit_score(data, features, outcome, cfg.model.random_seed, n_splits)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _best_delta_over_candidates(
    frame: pd.DataFrame,
    base_features: Sequence[str],
    candidates: Sequence[str],
    outcome: str,
    seed: int,
    n_splits: int,
) -> float:
    """The single best improvement any candidate achieves. One search pass."""
    base = _fit_score(frame, base_features, outcome, seed, n_splits)
    if not np.isfinite(base):
        return float("nan")
    best = -np.inf
    for candidate in candidates:
        score = _fit_score(frame, list(base_features) + [candidate], outcome, seed, n_splits)
        if np.isfinite(score):
            best = max(best, score - base)
    return best if np.isfinite(best) else float("nan")


def search_features(
    data: pd.DataFrame,
    base_features: Sequence[str] = ("log_BT_ratio", "age_years", "sex"),
    candidates: Sequence[str] | None = None,
    outcome: str = "declined",
    settings: Settings | None = None,
    min_coverage: float = 0.6,
    min_delta_auc: float = 0.02,
    n_permutations: int = 200,
    n_splits: int = 5,
) -> FeatureSearchResult:
    """Test candidate metrics and adopt one only if it genuinely helps.

    ``n_permutations`` controls the selection-level null. Lower it for a quick
    look; the default is the smallest count that gives a usable p-value at the
    0.05 threshold.
    """
    cfg = settings or SETTINGS
    seed = cfg.model.random_seed
    warnings_out: list[str] = []

    frame = data.copy()
    if "log_BT_ratio" not in frame.columns and "BT_ratio" in frame.columns:
        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
    if "participant_id" not in frame.columns:
        frame["participant_id"] = np.arange(len(frame))

    base = [f for f in base_features if f in frame.columns]
    missing_base = set(base_features) - set(base)
    if missing_base:
        warnings_out.append(f"Base predictors not present and dropped: {sorted(missing_base)}.")

    if candidates is None:
        from ..metrics import MetricRegistry

        registry = MetricRegistry(cfg)
        candidates = registry.candidate_features(exclude=set(base) | {outcome})
    candidates = [c for c in candidates if c in frame.columns and c not in base]

    # Protected attributes are refused before anything is fitted, so a
    # self-reported social category cannot become a predictor by accident.
    # dive/fairness.py explains why this is not merely a policy preference.
    from ..fairness import screen_features

    candidates, refused = screen_features(
        candidates,
        allow_protected=getattr(cfg, "allow_protected_features", False),
    )
    for name, why in refused:
        warnings_out.append(f"'{name}' excluded as a model feature: {why}")

    base_auc = _fit_score(frame, base, outcome, seed, n_splits)
    if not np.isfinite(base_auc):
        return FeatureSearchResult(
            base_features=base, adopted_features=base, base_auc=float("nan"),
            best_auc=float("nan"), best_candidate=None, delta_auc=float("nan"),
            selection_p_value=float("nan"),
            verdict="The baseline model could not be scored, so no metric can be tested against it.",
            warnings=warnings_out,
        )

    # --- evaluate each candidate -----------------------------------------
    evaluated: list[FeatureCandidate] = []
    usable_names: list[str] = []
    complete = frame.dropna(subset=base + [outcome])

    for name in candidates:
        coverage = float(complete[name].notna().mean()) if name in complete.columns else 0.0
        n_usable = int(complete[name].notna().sum()) if name in complete.columns else 0
        candidate = FeatureCandidate(name=name, coverage=coverage, n_usable=n_usable)

        if coverage < min_coverage:
            candidate.skip_reason = (
                f"measured on only {coverage * 100:.0f}% of eligible participants "
                f"(minimum {min_coverage * 100:.0f}%)"
            )
            evaluated.append(candidate)
            continue
        if frame[name].nunique(dropna=True) < 3:
            candidate.skip_reason = "too few distinct values to model"
            evaluated.append(candidate)
            continue

        score = _fit_score(frame, base + [name], outcome, seed, n_splits)
        candidate.auc = score
        candidate.delta_auc = score - base_auc if np.isfinite(score) else float("nan")
        candidate.evaluated = np.isfinite(score)
        if not candidate.evaluated:
            candidate.skip_reason = "could not be scored on these folds"
        else:
            usable_names.append(name)
        evaluated.append(candidate)

    scored = [c for c in evaluated if c.evaluated]
    if not scored:
        return FeatureSearchResult(
            base_features=base, adopted_features=base, base_auc=base_auc,
            best_auc=base_auc, best_candidate=None, delta_auc=float("nan"),
            selection_p_value=float("nan"),
            candidates=evaluated,
            verdict=(
                "No candidate metric had enough coverage to be tested. Nothing was adopted. "
                "More complete data on these metrics would be needed before they can help."
            ),
            warnings=warnings_out,
        )

    best = max(scored, key=lambda c: c.delta_auc)

    # --- selection-level permutation test --------------------------------
    # Shuffle the outcome and re-run the WHOLE search, so the null accounts for
    # having tried every candidate and kept the best.
    selection_p = float("nan")
    if n_permutations > 0 and len(usable_names) > 0:
        rng = np.random.default_rng(seed + 991)
        permuted_frame = frame.copy()
        null_deltas: list[float] = []
        outcome_values = frame[outcome].to_numpy()

        for _ in range(n_permutations):
            permuted_frame[outcome] = rng.permutation(outcome_values)
            delta = _best_delta_over_candidates(
                permuted_frame, base, usable_names, outcome, seed, n_splits
            )
            if np.isfinite(delta):
                null_deltas.append(delta)

        if null_deltas:
            null = np.asarray(null_deltas)
            selection_p = float((np.sum(null >= best.delta_auc) + 1) / (len(null) + 1))
        else:
            warnings_out.append("The selection-level null could not be built; p-value unavailable.")

    # --- adoption decision -----------------------------------------------
    adopted = list(base)
    if best.delta_auc < min_delta_auc:
        verdict = (
            f"No metric was adopted. The best candidate ({best.name}) changed cross-validated "
            f"AUC by {best.delta_auc:+.3f}, below the {min_delta_auc:.3f} threshold needed "
            "to count as an improvement."
        )
    elif not np.isfinite(selection_p):
        verdict = (
            f"{best.name} improved AUC by {best.delta_auc:+.3f}, but the selection-level "
            "null could not be computed, so the improvement cannot be distinguished from "
            "the effect of having tried several metrics. Nothing was adopted."
        )
    elif selection_p >= 0.05:
        verdict = (
            f"No metric was adopted. {best.name} raised AUC by {best.delta_auc:+.3f}, but "
            f"searching over {len(usable_names)} metrics found an improvement that large "
            f"{selection_p * 100:.0f}% of the time on shuffled outcomes. That is what "
            "searching does, not what this metric does."
        )
    else:
        adopted = list(base) + [best.name]
        verdict = (
            f"{best.name} was adopted. It raised cross-validated AUC from {base_auc:.3f} to "
            f"{best.auc:.3f} ({best.delta_auc:+.3f}), and an improvement that large arose on "
            f"shuffled outcomes only {selection_p * 100:.1f}% of the time even after "
            f"accounting for having tested {len(usable_names)} metrics."
        )
        from ..metrics import MetricRegistry

        MetricRegistry(cfg).mark_adopted(best.name, "decline")

    return FeatureSearchResult(
        base_features=base,
        adopted_features=adopted,
        base_auc=base_auc,
        best_auc=best.auc,
        best_candidate=best.name if adopted != base else None,
        delta_auc=best.delta_auc,
        selection_p_value=selection_p,
        candidates=evaluated,
        verdict=verdict,
        n_permutations=n_permutations,
        warnings=warnings_out,
    )
