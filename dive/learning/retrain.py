"""Gated retraining.

Retraining is only worth doing if it makes the model better, and the only
honest way to know that is to compare champion and challenger on the *same*
held-out participants that neither was fitted on.

The promotion gate has three conditions, all of which must hold:

1. The challenger beats the champion on the held-out metric by more than
   ``min_improvement`` - not by a hair that would vanish on a reshuffle.
2. The improvement survives a paired bootstrap, so it is not one lucky split.
3. The challenger still passes the null-safety checks: a decline model must
   beat its own permutation test before it can be promoted at all.

Condition 3 is the important one. Without it, a model that has learned nothing
can still be promoted for scoring marginally better noise than its predecessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import SETTINGS, Settings
from ..validation import participant_aware_folds
from .registry import ModelRegistry, ModelVersion

__all__ = ["RetrainOutcome", "retrain_and_evaluate", "detect_drift"]


@dataclass
class RetrainOutcome:
    model_kind: str
    challenger: ModelVersion | None
    champion: ModelVersion | None
    promoted: bool
    reason: str
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.promoted:
            return f"New {self.model_kind} model promoted: {self.reason}"
        return f"Existing {self.model_kind} model retained: {self.reason}"


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

def detect_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    columns: Sequence[str] | None = None,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Compare incoming data against the reference distribution per variable.

    Uses a two-sample Kolmogorov-Smirnov test plus a standardised mean
    difference. Drift is not automatically bad - a new cohort may legitimately
    differ - but it must be surfaced, because a model trained on one
    distribution and applied to another will be confidently wrong.
    """
    from scipy import stats

    columns = columns or [
        c for c in reference.columns
        if c in incoming.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]

    rows = []
    for column in columns:
        left = pd.to_numeric(reference[column], errors="coerce").dropna()
        right = pd.to_numeric(incoming[column], errors="coerce").dropna()
        if len(left) < 10 or len(right) < 10:
            continue
        statistic, p_value = stats.ks_2samp(left, right)
        pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
        smd = (right.mean() - left.mean()) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "variable": column,
                "n_reference": len(left),
                "n_incoming": len(right),
                "reference_mean": float(left.mean()),
                "incoming_mean": float(right.mean()),
                "std_mean_diff": float(smd),
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": bool(p_value < alpha and abs(smd) > 0.2),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("p_value").reset_index(drop=True) if not frame.empty else frame


# --------------------------------------------------------------------------
# Held-out comparison
# --------------------------------------------------------------------------

def _holdout_auc(model, data: pd.DataFrame, features: Sequence[str]) -> tuple[float, np.ndarray, np.ndarray]:
    frame = data.copy()
    if "log_BT_ratio" not in frame.columns and "BT_ratio" in frame.columns:
        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
    usable = [f for f in features if f in frame.columns]
    frame = frame.dropna(subset=usable + ["declined"])
    if frame.empty or frame["declined"].nunique() < 2:
        return float("nan"), np.array([]), np.array([])

    X = frame[usable].to_numpy(float)
    y = frame["declined"].to_numpy(int)
    probability = model.pipeline_.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probability)), y, probability


def _paired_bootstrap_auc_difference(
    y: np.ndarray,
    p_challenger: np.ndarray,
    p_champion: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap the AUC difference. Returns ``(mean_difference, p_worse)``."""
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        differences.append(
            roc_auc_score(y[idx], p_challenger[idx]) - roc_auc_score(y[idx], p_champion[idx])
        )
    if not differences:
        return float("nan"), float("nan")
    differences = np.array(differences)
    return float(differences.mean()), float((differences <= 0).mean())


# --------------------------------------------------------------------------
# Retraining
# --------------------------------------------------------------------------

def retrain_and_evaluate(
    substudy: pd.DataFrame,
    wide: pd.DataFrame | None = None,
    features: Sequence[str] = ("log_BT_ratio", "age_years", "sex"),
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    min_improvement: float = 0.02,
    holdout_fraction: float = 0.25,
    data_batches: Sequence[str] | None = None,
    notes: str = "",
) -> RetrainOutcome:
    """Fit a challenger decline model and promote it only if it earns it."""
    from ..decline_model import fit_decline_model

    cfg = settings or SETTINGS
    store = registry or ModelRegistry(cfg)
    warnings_out: list[str] = []

    frame = substudy.copy()
    if "log_BT_ratio" not in frame.columns and "BT_ratio" in frame.columns:
        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
    usable = [f for f in features if f in frame.columns]
    frame = frame.dropna(subset=usable + ["declined"])

    if len(frame) < cfg.model.abstain_min_training_n:
        return RetrainOutcome(
            model_kind="decline",
            challenger=None,
            champion=store.champion("decline"),
            promoted=False,
            reason=(
                f"Only {len(frame)} complete cases available - below the minimum of "
                f"{cfg.model.abstain_min_training_n}. Retraining was not attempted."
            ),
            warnings=warnings_out,
        )

    # Participant-disjoint holdout, so champion and challenger are judged on
    # people neither of them has seen.
    folds = participant_aware_folds(
        frame["participant_id"], n_splits=max(2, int(round(1 / holdout_fraction))),
        seed=cfg.model.random_seed,
    )
    train_idx, holdout_idx = folds[0]
    train = frame.iloc[train_idx]
    holdout = frame.iloc[holdout_idx]

    challenger_model = fit_decline_model(
        train, wide, features=features, settings=cfg, run_sensitivity=False
    )
    challenger_auc, y, p_challenger = _holdout_auc(challenger_model, holdout, features)
    result = challenger_model.result_

    challenger = store.register(
        challenger_model,
        model_kind="decline",
        metrics={
            "auc_cv": result.auc_cv,
            "auc_holdout": challenger_auc,
            "permutation_p": result.auc_permutation_p,
            "brier": result.brier,
        },
        n_training=len(train),
        n_participants=int(train["participant_id"].nunique()),
        features=list(usable),
        data_batches=list(data_batches or []),
        notes=notes,
    )

    champion = store.champion("decline")

    # --- Gate 3 first: a model that cannot beat chance is never promoted ---
    if cfg.strict_evidence_mode and not result.is_predictive:
        return RetrainOutcome(
            model_kind="decline",
            challenger=challenger,
            champion=champion,
            promoted=False,
            reason=(
                f"The challenger does not beat chance (permutation p = "
                f"{result.auc_permutation_p:.3f}). It will not be promoted regardless of "
                "how it compares to the incumbent - a better score on noise is still noise."
            ),
            warnings=warnings_out,
        )

    if champion is None:
        promoted = store.promote(
            challenger.version_id, reason="first validated model of its kind"
        )
        return RetrainOutcome(
            model_kind="decline",
            challenger=promoted,
            champion=None,
            promoted=True,
            reason=(
                f"No incumbent existed and the challenger passes validation "
                f"(holdout AUC {challenger_auc:.3f}, permutation p = "
                f"{result.auc_permutation_p:.3f})."
            ),
            warnings=warnings_out,
        )

    # --- Gates 1 and 2: beat the incumbent, and mean it -------------------
    try:
        champion_model = store.load_model(champion.version_id)
        champion_auc, _, p_champion = _holdout_auc(champion_model, holdout, features)
    except (FileNotFoundError, AttributeError, KeyError) as exc:
        warnings_out.append(f"Champion artefact could not be scored ({exc}); keeping incumbent.")
        champion_auc, p_champion = float("nan"), np.array([])

    comparison = pd.DataFrame(
        [
            {"model": "champion", "version": champion.version_id, "holdout_auc": champion_auc,
             "cv_auc": champion.metrics.get("auc_cv", np.nan)},
            {"model": "challenger", "version": challenger.version_id, "holdout_auc": challenger_auc,
             "cv_auc": result.auc_cv},
        ]
    )

    if not np.isfinite(champion_auc) or len(p_champion) != len(p_challenger):
        return RetrainOutcome(
            "decline", challenger, champion, False,
            "The incumbent could not be evaluated on the same holdout, so no like-for-like "
            "comparison was possible. Incumbent retained.",
            comparison, warnings_out,
        )

    improvement = challenger_auc - champion_auc
    mean_difference, p_worse = _paired_bootstrap_auc_difference(
        y, p_challenger, p_champion, seed=cfg.model.random_seed
    )

    if improvement < min_improvement:
        reason = (
            f"Challenger holdout AUC {challenger_auc:.3f} vs incumbent {champion_auc:.3f} "
            f"({improvement:+.3f}) does not clear the {min_improvement:.3f} improvement "
            "threshold. Incumbent retained."
        )
    elif np.isfinite(p_worse) and p_worse > 0.05:
        reason = (
            f"Challenger is nominally better ({improvement:+.3f} AUC) but the paired "
            f"bootstrap puts it behind the incumbent in {p_worse * 100:.0f}% of resamples. "
            "That is one lucky split, not an improvement. Incumbent retained."
        )
    else:
        store.promote(
            challenger.version_id,
            reason=f"holdout AUC {challenger_auc:.3f} vs {champion_auc:.3f} ({improvement:+.3f})",
        )
        return RetrainOutcome(
            "decline", challenger, champion, True,
            f"Challenger improved holdout AUC by {improvement:+.3f} "
            f"({champion_auc:.3f} -> {challenger_auc:.3f}) and the improvement held in "
            f"{(1 - p_worse) * 100:.0f}% of bootstrap resamples.",
            comparison, warnings_out,
        )

    return RetrainOutcome("decline", challenger, champion, False, reason, comparison, warnings_out)
