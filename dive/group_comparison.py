"""Group comparison of the B/T ratio across faller subgroups.

Two tests, reported side by side:

* **Kruskal-Wallis** - non-parametric, the PORAF default for continuous
  variables that are not normally distributed.
* **Gender-adjusted ANCOVA** - Type II sums of squares, with an optional
  subgroup x sex interaction, because the trans-signalling literature reports
  sex-differential sIL-6R and sgp130 concentrations.

The module reports effect sizes alongside p-values and produces a
plain-language verdict. On null synthetic data the verdict must read as "no
evidence of a difference" - ``assert_null_on_synthetic`` enforces this as a
pipeline sanity check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import SETTINGS, Settings

__all__ = [
    "GroupComparisonResult",
    "kruskal_wallis_bt",
    "ancova_bt",
    "run_group_comparison",
    "assert_null_on_synthetic",
]


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class GroupComparisonResult:
    outcome: str
    group_variable: str
    descriptives: pd.DataFrame
    kruskal: dict
    ancova: pd.DataFrame
    ancova_interaction: pd.DataFrame | None
    verdict: str
    technical_note: str
    n_total: int
    n_groups: int
    warnings: list[str] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        p = self.kruskal.get("p_value", np.nan)
        return bool(np.isfinite(p) and p < 0.05)

    def summary_table(self) -> pd.DataFrame:
        """A single presentable table combining both tests."""
        rows = [
            {
                "Test": "Kruskal-Wallis",
                "Statistic": f"H = {self.kruskal['statistic']:.3f}",
                "df": self.kruskal["df"],
                "p": _format_p(self.kruskal["p_value"]),
                "Effect size": f"eps^2 = {self.kruskal['epsilon_squared']:.3f}",
            }
        ]
        if not self.ancova.empty and self.group_variable in self.ancova.index:
            row = self.ancova.loc[self.group_variable]
            rows.append(
                {
                    "Test": "ANCOVA (sex-adjusted)",
                    "Statistic": f"F = {row['F']:.3f}",
                    "df": f"{int(row['df'])}, {int(self.ancova.loc['Residual', 'df'])}",
                    "p": _format_p(row["p"]),
                    "Effect size": f"partial eta^2 = {row['partial_eta_sq']:.3f}",
                }
            )
        return pd.DataFrame(rows)


def _format_p(p: float) -> str:
    """PORAF reporting convention: p < 0.001 rather than a long decimal."""
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def kruskal_wallis_bt(
    data: pd.DataFrame,
    outcome: str = "BT_ratio",
    group: str = "subgroup",
    min_per_group: int = 3,
) -> tuple[dict, list[str]]:
    """Kruskal-Wallis across groups, with epsilon-squared effect size."""
    warnings: list[str] = []
    frame = data[[outcome, group]].dropna()

    counts = frame[group].value_counts()
    too_small = counts[counts < min_per_group]
    if not too_small.empty:
        warnings.append(
            "Groups with fewer than "
            f"{min_per_group} observations excluded from Kruskal-Wallis: "
            + ", ".join(f"{k} (n={v})" for k, v in too_small.items())
        )
        frame = frame[frame[group].isin(counts[counts >= min_per_group].index)]

    groups = [g[outcome].to_numpy() for _, g in frame.groupby(group, observed=True)]
    k = len(groups)
    n = len(frame)

    if k < 2 or n < 3:
        return (
            {"statistic": np.nan, "p_value": np.nan, "df": max(k - 1, 0),
             "epsilon_squared": np.nan, "n": n, "k": k},
            warnings + ["Not enough groups or observations for Kruskal-Wallis."],
        )

    statistic, p_value = stats.kruskal(*groups)
    # Epsilon-squared: the standard KW effect size, bounded 0-1.
    epsilon_sq = (statistic - k + 1) / (n - k) if n > k else np.nan

    return (
        {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df": k - 1,
            "epsilon_squared": float(max(epsilon_sq, 0.0)) if np.isfinite(epsilon_sq) else np.nan,
            "n": int(n),
            "k": int(k),
        },
        warnings,
    )


def ancova_bt(
    data: pd.DataFrame,
    outcome: str = "log_BT_ratio",
    group: str = "subgroup",
    covariate: str = "sex_label",
    interaction: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Type II ANCOVA of ``outcome`` on ``group``, adjusted for ``covariate``.

    The B/T ratio is right-skewed, so the ANCOVA is fitted on the log scale by
    default - ANCOVA assumes normal residuals and fitting the raw ratio would
    violate that assumption on exactly the tail that matters.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    notes: list[str] = []
    needed = [c for c in (outcome, group, covariate) if c in data.columns]
    if len(needed) < 3:
        return pd.DataFrame(), ["ANCOVA skipped: required columns missing."]

    frame = data[needed].dropna().copy()
    frame[group] = frame[group].astype(str)
    frame[covariate] = frame[covariate].astype(str)

    # Drop singleton cells - they contribute no within-cell variance.
    counts = frame[group].value_counts()
    keep = counts[counts >= 3].index
    if len(keep) < len(counts):
        notes.append(
            "ANCOVA: groups with n<3 dropped ("
            + ", ".join(f"{k} (n={v})" for k, v in counts[counts < 3].items())
            + ")"
        )
        frame = frame[frame[group].isin(keep)]

    if frame[group].nunique() < 2:
        return pd.DataFrame(), notes + ["ANCOVA skipped: fewer than two usable groups."]

    # An empty subgroup x covariate cell makes the design rank-deficient, which
    # statsmodels will fit anyway while warning that the parameters are not
    # uniquely determined. With four faller subgroups and a small substudy that
    # is a live possibility, so check for it and report it as a limitation
    # rather than letting a raw warning escape to the user.
    formula = f"Q('{outcome}') ~ C(Q('{group}')) {'*' if interaction else '+'} C(Q('{covariate}'))"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=frame).fit()

    design_rank = int(np.linalg.matrix_rank(model.model.exog))
    n_params = model.model.exog.shape[1]
    if design_rank < n_params:
        empty = (
            frame.groupby([group, covariate], observed=True).size().unstack(fill_value=0)
        )
        empty_cells = [
            f"{row} x {column}"
            for row in empty.index for column in empty.columns if empty.loc[row, column] == 0
        ]
        message = (
            f"ANCOVA design is rank-deficient ({design_rank} of {n_params} parameters "
            "identifiable): at least one subgroup-by-sex cell is empty"
        )
        if empty_cells:
            message += " (" + ", ".join(empty_cells[:4]) + ")"
        message += ". The adjusted F is not uniquely determined - prefer the Kruskal-Wallis result."
        analysis_warnings_rank = [message]
    else:
        analysis_warnings_rank = []

    table = sm.stats.anova_lm(model, typ=2)

    # Rename the mangled patsy terms back to readable names.
    rename: dict[str, str] = {}
    for term in table.index:
        if term == "Residual":
            continue
        label = term.replace(f"C(Q('{group}'))", group).replace(f"C(Q('{covariate}'))", covariate)
        rename[term] = label
    table = table.rename(index=rename)
    table = table.rename(columns={"sum_sq": "sum_sq", "PR(>F)": "p"})

    ss_resid = float(table.loc["Residual", "sum_sq"])
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    table.loc["Residual", "partial_eta_sq"] = np.nan
    table.loc["Residual", "F"] = np.nan
    table.loc["Residual", "p"] = np.nan

    # Levene's test for the homogeneity-of-variance assumption.
    groups = [g[outcome].to_numpy() for _, g in frame.groupby(group, observed=True)]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        _, levene_p = stats.levene(*groups)
        if levene_p < 0.05:
            notes.append(
                f"Levene's test p={_format_p(levene_p)}: group variances differ; "
                "treat the ANCOVA F as approximate and prefer the Kruskal-Wallis result."
            )

    return table, analysis_warnings_rank + notes


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _describe(data: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    """Median (IQR) and mean (SD) per group - PORAF reports both."""
    frame = data[[outcome, group]].dropna()
    rows = []
    for name, chunk in frame.groupby(group, observed=True):
        values = chunk[outcome].to_numpy()
        q1, q3 = np.percentile(values, [25, 75])
        rows.append(
            {
                "Group": name,
                "n": len(values),
                "Median (IQR)": f"{np.median(values):.3f} ({q1:.3f}-{q3:.3f})",
                "Mean (SD)": f"{values.mean():.3f} ({values.std(ddof=1):.3f})",
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def _verdict(kruskal: dict, ancova: pd.DataFrame, group: str, outcome_label: str) -> tuple[str, str]:
    """Plain-language verdict first, technical detail second."""
    p_kw = kruskal.get("p_value", np.nan)
    eps = kruskal.get("epsilon_squared", np.nan)

    p_anc = np.nan
    if not ancova.empty and group in ancova.index:
        p_anc = float(ancova.loc[group, "p"])

    if not np.isfinite(p_kw):
        return (
            f"There was not enough data to test whether {outcome_label} differs between groups.",
            "Test not run: insufficient groups or observations.",
        )

    significant = p_kw < 0.05
    adjusted_significant = np.isfinite(p_anc) and p_anc < 0.05

    if not significant and not adjusted_significant:
        verdict = (
            f"No evidence that {outcome_label} differs between the faller subgroups. "
            f"The groups look the same on this measure (p = {_format_p(p_kw)})."
        )
    elif significant and adjusted_significant:
        magnitude = "small" if eps < 0.06 else "moderate" if eps < 0.14 else "large"
        verdict = (
            f"{outcome_label} does differ between the faller subgroups, and the difference "
            f"survives adjustment for sex (p = {_format_p(p_kw)} unadjusted, "
            f"p = {_format_p(p_anc)} adjusted). The effect is {magnitude}."
        )
    elif significant and not adjusted_significant:
        verdict = (
            f"{outcome_label} differed between subgroups before adjustment "
            f"(p = {_format_p(p_kw)}), but the difference did not survive adjustment for sex "
            f"(p = {_format_p(p_anc)}). This is most likely a sex-composition effect, "
            "not a subgroup effect."
        )
    else:
        verdict = (
            f"{outcome_label} showed no unadjusted difference (p = {_format_p(p_kw)}), but a "
            f"difference emerged after adjusting for sex (p = {_format_p(p_anc)}). "
            "Interpret cautiously - suppression effects in small samples are unstable."
        )

    technical = (
        f"Kruskal-Wallis H({kruskal['df']}) = {kruskal['statistic']:.3f}, "
        f"p = {_format_p(p_kw)}, epsilon^2 = {eps:.3f}, n = {kruskal['n']} across "
        f"{kruskal['k']} groups."
    )
    if np.isfinite(p_anc):
        technical += (
            f" Sex-adjusted ANCOVA on log(B/T): F = {ancova.loc[group, 'F']:.3f}, "
            f"p = {_format_p(p_anc)}, partial eta^2 = {ancova.loc[group, 'partial_eta_sq']:.3f}."
        )
    return verdict, technical


def run_group_comparison(
    data: pd.DataFrame,
    outcome: str = "BT_ratio",
    group: str = "subgroup",
    covariate: str = "sex_label",
    test_interaction: bool = True,
    settings: Settings | None = None,
) -> GroupComparisonResult:
    """Run the full group comparison and return every component of the result."""
    cfg = settings or SETTINGS
    frame = data.copy()

    if outcome == "BT_ratio" and "log_BT_ratio" not in frame.columns:
        frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))

    kruskal, kw_warnings = kruskal_wallis_bt(frame, outcome=outcome, group=group)

    ancova_outcome = "log_BT_ratio" if outcome == "BT_ratio" else outcome
    ancova, anc_warnings = ancova_bt(
        frame, outcome=ancova_outcome, group=group, covariate=covariate, interaction=False
    )

    interaction_table: pd.DataFrame | None = None
    int_warnings: list[str] = []
    if test_interaction:
        interaction_table, int_warnings = ancova_bt(
            frame, outcome=ancova_outcome, group=group, covariate=covariate, interaction=True
        )
        if interaction_table is not None and not interaction_table.empty:
            term = [i for i in interaction_table.index if ":" in str(i)]
            if term:
                p_int = float(interaction_table.loc[term[0], "p"])
                if np.isfinite(p_int) and p_int >= 0.05:
                    int_warnings.append(
                        f"Subgroup x sex interaction not significant (p = {_format_p(p_int)}); "
                        "the additive sex-adjusted model is reported as primary."
                    )

    outcome_label = "the B/T ratio" if outcome == "BT_ratio" else outcome.replace("_", " ")
    verdict, technical = _verdict(kruskal, ancova, group, outcome_label)

    descriptives = _describe(frame, outcome, group)

    return GroupComparisonResult(
        outcome=outcome,
        group_variable=group,
        descriptives=descriptives,
        kruskal=kruskal,
        ancova=ancova,
        ancova_interaction=interaction_table,
        verdict=verdict,
        technical_note=technical,
        n_total=int(kruskal.get("n", 0)),
        n_groups=int(kruskal.get("k", 0)),
        warnings=kw_warnings + anc_warnings + int_warnings,
    )


def assert_null_on_synthetic(result: GroupComparisonResult, alpha: float = 0.05) -> None:
    """Pipeline sanity check.

    On synthetic data generated with a null B/T effect, the group comparison
    must NOT find a significant difference. A significant result here means
    the pipeline is manufacturing signal, not that the biology is real.
    """
    p = result.kruskal.get("p_value", np.nan)
    if np.isfinite(p) and p < alpha:
        raise AssertionError(
            f"Group comparison returned p = {p:.4f} (< {alpha}) on synthetic data "
            "generated with a NULL B/T effect. The pipeline is producing signal where "
            "none was simulated - investigate before trusting any real-data result."
        )
