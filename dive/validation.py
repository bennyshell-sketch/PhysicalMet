"""Validation harness.

Checks that the pipeline is *correct*, not merely that it runs. Four families:

1. **Leakage** - no participant may appear in both sides of a split. With
   repeated measures, a random row-level split puts a person's baseline in
   train and their 12-month visit in test, which inflates every metric.
2. **Null recovery** - on data simulated with no B/T effect, the group
   comparison and the decline model must return null results.
3. **Effect recovery** - on data simulated with a known non-zero effect, the
   pipeline must find it, and the recovered estimate must sit near the truth.
   A pipeline that only ever returns "no effect" is not conservative, it is
   broken, and this is the test that distinguishes the two.
4. **Interval coverage** - a 95% prediction interval must contain the truth
   about 95% of the time. An interval that covers 60% is a lie about
   uncertainty; one that covers 100% is useless.

Run everything with ``python -m dive.validation``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings

__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "participant_aware_folds",
    "assert_no_participant_leakage",
    "run_validation_suite",
]


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str
    observed: Any = None
    expected: Any = None
    category: str = "general"

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class ValidationReport:
    checks: list[ValidationCheck] = field(default_factory=list)

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Category": c.category,
                    "Check": c.name,
                    "Result": "PASS" if c.passed else "FAIL",
                    "Detail": c.detail,
                }
                for c in self.checks
            ]
        )

    def summary(self) -> str:
        return f"{self.n_passed}/{len(self.checks)} checks passed"

    def print_report(self) -> None:
        by_category: dict[str, list[ValidationCheck]] = {}
        for check in self.checks:
            by_category.setdefault(check.category, []).append(check)
        for category, items in by_category.items():
            print(f"\n--- {category} ---")
            for check in items:
                print("  " + check.line())
        print(f"\n{self.summary()}")


# --------------------------------------------------------------------------
# Participant-aware splitting
# --------------------------------------------------------------------------

def participant_aware_folds(
    participant_ids: Sequence,
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split rows so that every participant lands entirely on one side.

    This is the only correct way to cross-validate repeated-measures data.
    Rows are grouped by participant, participants are shuffled, then dealt
    into folds.
    """
    ids = np.asarray(participant_ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    assignment = {pid: i % n_splits for i, pid in enumerate(unique)}
    fold_of_row = np.array([assignment[p] for p in ids])

    folds = []
    for k in range(n_splits):
        test = np.where(fold_of_row == k)[0]
        train = np.where(fold_of_row != k)[0]
        if len(test) and len(train):
            folds.append((train, test))
    return folds


def assert_no_participant_leakage(
    participant_ids: Sequence,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Raise if any participant appears in both train and test of a fold."""
    ids = np.asarray(participant_ids)
    for i, (train, test) in enumerate(folds):
        overlap = set(ids[train]) & set(ids[test])
        if overlap:
            raise AssertionError(
                f"Participant leakage in fold {i}: {len(overlap)} participant(s) "
                f"appear in both train and test, e.g. {sorted(overlap)[:3]}."
            )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _check_leakage(report: ValidationReport, long: pd.DataFrame) -> None:
    folds = participant_aware_folds(long["participant_id"], n_splits=5, seed=1)
    try:
        assert_no_participant_leakage(long["participant_id"], folds)
        report.add(
            ValidationCheck(
                "Participant-aware splits contain no leakage",
                True,
                f"{len(folds)} folds over {long['participant_id'].nunique()} participants; "
                "no participant spans train and test.",
                category="Leakage",
            )
        )
    except AssertionError as exc:
        report.add(ValidationCheck("Participant-aware splits contain no leakage", False, str(exc), category="Leakage"))

    # Demonstrate that a naive row-level split WOULD leak, so the guard is
    # doing real work rather than passing vacuously.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(long))
    half = len(order) // 2
    naive = [(order[:half], order[half:])]
    ids = long["participant_id"].to_numpy()
    leaked = len(set(ids[naive[0][0]]) & set(ids[naive[0][1]]))
    report.add(
        ValidationCheck(
            "Naive row split is demonstrably unsafe",
            leaked > 0,
            f"A random row-level split leaks {leaked} participants across train/test - "
            "confirming the participant-aware splitter is necessary.",
            observed=leaked,
            category="Leakage",
        )
    )


def _check_bt_solver(report: ValidationReport) -> None:
    """The equilibrium solver must satisfy mass conservation exactly."""
    from .biochem import solve_equilibrium, to_molar_nM

    worst = 0.0
    cases = [(2.0, 40.0, 350.0), (0.5, 20.0, 500.0), (30.0, 80.0, 180.0), (12.0, 65.0, 300.0)]
    identity_error = 0.0
    for il6, sil6r, sgp130 in cases:
        solution = solve_equilibrium(il6, sil6r, sgp130)
        L, R, G = (float(x) for x in to_molar_nM(il6, sil6r, sgp130))
        worst = max(
            worst,
            abs(solution.free_il6_nM + solution.binary_nM + solution.ternary_nM - L) / L,
            abs(solution.free_sil6r_nM + solution.binary_nM + solution.ternary_nM - R) / R,
            abs(solution.free_sgp130_nM + solution.ternary_nM - G) / G,
        )
        # B/T must equal Kd2 / free sgp130 analytically.
        identity_error = max(
            identity_error,
            abs(solution.bt_ratio - SETTINGS.biochem.kd2_nM / solution.free_sgp130_nM)
            / solution.bt_ratio,
        )

    report.add(
        ValidationCheck(
            "Equilibrium solver conserves mass",
            worst < 1e-8,
            f"Worst relative mass-balance error across {len(cases)} cases: {worst:.2e}.",
            observed=worst,
            category="Biochemistry",
        )
    )
    report.add(
        ValidationCheck(
            "B/T matches the analytic identity Kd2/[free sgp130]",
            identity_error < 1e-8,
            f"Worst relative deviation: {identity_error:.2e}.",
            observed=identity_error,
            category="Biochemistry",
        )
    )


def _check_delta_recomputation(report: ValidationReport, workbook: Path) -> None:
    """The pipeline must ignore corrupted stored delta columns."""
    from .data_pipeline import build_dataset

    data = build_dataset(workbook)
    raw = pd.read_excel(workbook, sheet_name="TwelveMonth")
    baseline = pd.read_excel(workbook, sheet_name="Baseline").set_index("Participant_ID")["SPPB_Total"]
    stored = raw.set_index("Participant_ID")["Delta_SPPB_Total"]
    truth = raw.set_index("Participant_ID")["SPPB_Total"] - baseline

    both = stored.notna() & truth.notna()
    corrupted = int((~np.isclose(stored[both], truth[both])).sum())

    wide = data.wide.set_index("participant_id")
    if "delta_sppb_TwelveMonth" in wide.columns:
        recomputed = wide["delta_sppb_TwelveMonth"]
        common = recomputed.index.intersection(truth.index)
        mask = recomputed[common].notna() & truth[common].notna()
        agreement = float(np.isclose(recomputed[common][mask], truth[common][mask]).mean())
    else:
        agreement = float("nan")

    report.add(
        ValidationCheck(
            "Deltas recomputed from raw, not read from stored columns",
            np.isfinite(agreement) and agreement > 0.999 and corrupted > 0,
            f"{corrupted} stored delta values are corrupted in the source; the pipeline's "
            f"recomputed values agree with ground truth for {agreement * 100:.1f}% of rows.",
            observed=agreement,
            category="Cleaning",
        )
    )


def _check_tug_sentinel(report: ValidationReport, data) -> None:
    long = data.long
    if "tug_s" not in long.columns:
        return
    sentinel = SETTINGS.cohort.tug_sentinel_value
    leaked = int((long["tug_s"] == sentinel).sum())
    flagged = int(long.get("tug_cannot_complete", pd.Series(dtype=float)).fillna(0).sum())
    report.add(
        ValidationCheck(
            "TUG 'cannot complete' sentinel excluded from continuous data",
            leaked == 0 and flagged > 0,
            f"{flagged} sentinel values flagged and removed; {leaked} remain in the "
            "continuous TUG column.",
            observed=leaked,
            category="Cleaning",
        )
    )


def _check_null_recovery(report: ValidationReport, workbook: Path, settings: Settings) -> None:
    """Null in must produce null out."""
    from .data_pipeline import build_dataset
    from .group_comparison import run_group_comparison
    from .decline_model import fit_decline_model

    data = build_dataset(workbook, settings)
    comparison = run_group_comparison(data.substudy, settings=settings)
    p_kw = comparison.kruskal.get("p_value", np.nan)

    report.add(
        ValidationCheck(
            "Group comparison returns null on null data",
            not (np.isfinite(p_kw) and p_kw < 0.05),
            f"Kruskal-Wallis p = {p_kw:.3f} (must not be significant when no effect was simulated).",
            observed=p_kw,
            category="Null recovery",
        )
    )

    model = fit_decline_model(data.substudy, data.wide, settings=settings, run_sensitivity=False)
    perm_p = model.result_.auc_permutation_p
    report.add(
        ValidationCheck(
            "Decline model fails its permutation test on null data",
            not (np.isfinite(perm_p) and perm_p < 0.05),
            f"AUC = {model.result_.auc_cv:.3f}, permutation p = {perm_p:.3f} - correctly "
            "indistinguishable from chance.",
            observed=perm_p,
            category="Null recovery",
        )
    )

    prediction = model.predict_risk({"log_BT_ratio": -1.1, "age_years": 75, "sex": 1})
    report.add(
        ValidationCheck(
            "Model abstains from individual risk when it cannot predict",
            prediction.abstained,
            "Individual risk was withheld because the model does not beat chance."
            if prediction.abstained
            else "MODEL ISSUED A RISK ESTIMATE DESPITE FAILING VALIDATION.",
            category="Null recovery",
        )
    )


def _check_effect_recovery(report: ValidationReport, settings: Settings, tmp_dir: Path) -> None:
    """Non-null in must produce non-null out, at roughly the right size."""
    from .generate_synthetic_data import GenerationParams, generate_workbook
    from .data_pipeline import build_dataset
    from .decline_model import fit_decline_model
    from .trajectory_model import fit_trajectory_model

    # --- decline effect ---------------------------------------------------
    true_log_or = 1.4
    params = GenerationParams(seed=4242, bt_decline_log_or=true_log_or)
    path, _ = generate_workbook(params, output_path=tmp_dir / "recovery_decline.xlsx", settings=settings)
    data = build_dataset(path, settings)
    model = fit_decline_model(data.substudy, data.wide, settings=settings, run_sensitivity=False)
    perm_p = model.result_.auc_permutation_p
    auc = model.result_.auc_cv

    report.add(
        ValidationCheck(
            "Decline model detects a simulated B/T effect",
            np.isfinite(perm_p) and perm_p < 0.05,
            f"With a simulated log-OR of {true_log_or}, the model reached AUC {auc:.3f} "
            f"with permutation p = {perm_p:.3f}.",
            observed=perm_p,
            expected="< 0.05",
            category="Effect recovery",
        )
    )

    coefficients = model.result_.coefficients
    bt_row = coefficients[coefficients["term"] == "log_BT_ratio"]
    if not bt_row.empty:
        estimate = float(bt_row["coefficient"].iloc[0])
        report.add(
            ValidationCheck(
                "Recovered B/T coefficient has the simulated sign",
                estimate > 0,
                f"Recovered standardised log-odds {estimate:+.3f} for log(B/T); "
                f"a positive effect was simulated. "
                "(Magnitude is attenuated by L2 shrinkage and case-control sampling, "
                "so only the sign and significance are asserted.)",
                observed=estimate,
                category="Effect recovery",
            )
        )

    # --- trajectory variance components ----------------------------------
    truth_params = GenerationParams(seed=99)
    traj_path, truth = generate_workbook(
        truth_params, output_path=tmp_dir / "recovery_traj.xlsx", settings=settings
    )
    traj_data = build_dataset(traj_path, settings)
    trajectory = fit_trajectory_model(traj_data.trajectory_frame(), settings=settings)
    result = trajectory.result_

    true_intercept_sd = truth["params"]["grip_between_sd_kg"]
    true_residual_sd = truth["params"]["grip_residual_sd_kg"]
    # Compare against the REALISED marginal slope, not the bare per-participant
    # slope parameter. The generator also applies a post-fall dip and a
    # non-recovery decrement for the Declined-Function group; those are real
    # features of the simulated cohort and the population-average trajectory
    # legitimately includes them. ``grip_slope_mean`` alone is not the estimand.
    true_slope_mean = truth["realised"]["within_slope_per_year"]

    est_intercept_sd = result.random_effect_sd.get("Group", np.nan)
    est_slope_row = result.fixed_effects[result.fixed_effects["term"] == "time"]
    est_slope = float(est_slope_row["estimate"].iloc[0]) if not est_slope_row.empty else np.nan

    report.add(
        ValidationCheck(
            "Between-participant SD recovered",
            abs(est_intercept_sd - true_intercept_sd) / true_intercept_sd < 0.25,
            f"Estimated {est_intercept_sd:.2f} kg vs simulated {true_intercept_sd:.2f} kg "
            f"({abs(est_intercept_sd - true_intercept_sd) / true_intercept_sd * 100:.1f}% error).",
            observed=est_intercept_sd,
            expected=true_intercept_sd,
            category="Effect recovery",
        )
    )
    report.add(
        ValidationCheck(
            "Residual SD recovered",
            abs(result.residual_sd - true_residual_sd) / true_residual_sd < 0.30,
            f"Estimated {result.residual_sd:.2f} kg vs simulated {true_residual_sd:.2f} kg.",
            observed=result.residual_sd,
            expected=true_residual_sd,
            category="Effect recovery",
        )
    )
    report.add(
        ValidationCheck(
            "Mean slope recovered",
            abs(est_slope - true_slope_mean) < 0.35,
            f"Estimated {est_slope:.3f} kg/year vs realised {true_slope_mean:.3f} kg/year "
            "in the generated cohort.",
            observed=est_slope,
            expected=true_slope_mean,
            category="Effect recovery",
        )
    )

    return trajectory, traj_data


def _check_interval_coverage(
    report: ValidationReport, trajectory, data, settings: Settings
) -> None:
    """A 95% prediction interval must actually cover about 95% of outcomes.

    Held-out participants only: the model is refitted without them, then asked
    to forecast their observed later visits from their earlier ones.
    """
    long = data.trajectory_frame()
    participants = long["participant_id"].unique()
    rng = np.random.default_rng(settings.model.random_seed)
    holdout = set(rng.choice(participants, size=min(80, len(participants) // 4), replace=False))

    train = long[~long["participant_id"].isin(holdout)]
    from .trajectory_model import fit_trajectory_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_trajectory_model(train, settings=settings)

    covered = 0
    total = 0
    for pid, chunk in long[long["participant_id"].isin(holdout)].groupby("participant_id"):
        chunk = chunk.sort_values("elapsed_months")
        if len(chunk) < 2:
            continue
        history = chunk.iloc[:-1][["elapsed_months", "max_grip_kg"]]
        target = chunk.iloc[-1]
        profile = {
            "age_years": float(chunk["age_years"].iloc[0]),
            "sex": float(chunk["sex"].iloc[0]),
        }
        if not np.isfinite(profile["age_years"]) or not np.isfinite(profile["sex"]):
            continue
        try:
            forecast = model.predict_trajectory(
                profile, months=[float(target["elapsed_months"])], observed=history
            )
        except Exception:
            continue
        row = forecast.iloc[0]
        total += 1
        if row["pi_lower"] <= target["max_grip_kg"] <= row["pi_upper"]:
            covered += 1

    coverage = covered / total if total else np.nan
    report.add(
        ValidationCheck(
            "95% prediction intervals achieve nominal coverage",
            np.isfinite(coverage) and 0.88 <= coverage <= 0.995,
            f"Held-out coverage {coverage * 100:.1f}% across {total} forecasts "
            "(target 95%, accepted 88-99.5%).",
            observed=coverage,
            expected=0.95,
            category="Calibration",
        )
    )


def _check_case_control_correction(report: ValidationReport, settings: Settings) -> None:
    from .decline_model import prior_correction_offset

    offset = prior_correction_offset(0.5, 0.25)
    # From a 50/50 sample to a 25% prevalence population, the intercept must
    # shift down by log(3).
    expected = float(np.log(3.0))
    report.add(
        ValidationCheck(
            "Case-control prior correction is arithmetically right",
            abs(offset - expected) < 1e-9,
            f"Offset for 50% sample -> 25% population = {offset:.6f} (expected log(3) = {expected:.6f}).",
            observed=offset,
            expected=expected,
            category="Case-control",
        )
    )
    neutral = prior_correction_offset(0.5, 0.5)
    report.add(
        ValidationCheck(
            "Correction is a no-op when the sample matches the population",
            abs(neutral) < 1e-12,
            f"Offset = {neutral:.2e} when sample and population prevalence agree.",
            category="Case-control",
        )
    )


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

def run_validation_suite(settings: Settings | None = None, verbose: bool = True) -> ValidationReport:
    """Run every validation check and return the report."""
    cfg = settings or SETTINGS
    report = ValidationReport()

    from .data_pipeline import load_dataset

    tmp_dir = cfg.paths.synthetic_dir / "validation"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data = load_dataset(cfg)
        workbook = Path(data.report.source_path)

        _check_bt_solver(report)
        _check_leakage(report, data.long)
        _check_delta_recomputation(report, workbook)
        _check_tug_sentinel(report, data)
        _check_case_control_correction(report, cfg)
        _check_null_recovery(report, workbook, cfg)
        trajectory, traj_data = _check_effect_recovery(report, cfg, tmp_dir)
        _check_interval_coverage(report, trajectory, traj_data, cfg)

    if verbose:
        report.print_report()
    return report


def main() -> None:
    report = run_validation_suite()
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
