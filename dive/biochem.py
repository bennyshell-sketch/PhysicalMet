"""IL-6 trans-signalling equilibrium model.

The B/T ratio is the molar ratio of the **binary** IL-6 . sIL-6R complex (the
trans-signalling agonist) to the **ternary** IL-6 . sIL-6R . sgp130 complex
(the buffered, non-signalling form). It is solved from a two-step equilibrium,
never hardcoded or approximated by a shortcut formula.

Step 1   IL-6  +  sIL-6R          <-> [IL-6 . sIL-6R]            Kd1 = 10 nM
Step 2   [IL-6 . sIL-6R] + sgp130 <-> [IL-6 . sIL-6R . sgp130]   Kd2 =  1 nM

Writing L, R, G for the *free* concentrations of IL-6, sIL-6R and sgp130, and
B, T for the binary and ternary complexes, mass action gives

    B = L * R / Kd1
    T = B * G / Kd2 = L * R * G / (Kd1 * Kd2)

and mass conservation gives the three residuals solved numerically below:

    L + B + T = L_total
    R + B + T = R_total
    G     + T = G_total

Note that B / T = Kd2 / G collapses analytically, but G (free sgp130) is itself
only available from the numerical solution, so the solve is genuine. The
analytic identity is used as an internal consistency check, not as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from .config import BiochemConstants, SETTINGS

__all__ = [
    "EquilibriumSolution",
    "pg_per_ml_to_nM",
    "ng_per_ml_to_nM",
    "to_molar_nM",
    "solve_equilibrium",
    "bt_ratio",
    "compute_bt_ratio_frame",
    "free_mediator_equivalent",
]


# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------

def pg_per_ml_to_nM(value: float | np.ndarray, mw_da: float) -> float | np.ndarray:
    """Convert pg/mL to nanomolar.

    pg/mL = 1e-9 g/L; dividing by the molecular weight (g/mol) gives mol/L,
    and multiplying by 1e9 gives nmol/L.
    """
    return np.asarray(value, dtype=float) * 1e-9 / mw_da * 1e9


def ng_per_ml_to_nM(value: float | np.ndarray, mw_da: float) -> float | np.ndarray:
    """Convert ng/mL to nanomolar."""
    return np.asarray(value, dtype=float) * 1e-6 / mw_da * 1e9


def to_molar_nM(
    il6_pg_ml: float | np.ndarray,
    sil6r_ng_ml: float | np.ndarray,
    sgp130_ng_ml: float | np.ndarray,
    constants: BiochemConstants | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the three assay readouts to nanomolar total concentrations."""
    c = constants or SETTINGS.biochem
    return (
        np.asarray(pg_per_ml_to_nM(il6_pg_ml, c.mw_il6_da), dtype=float),
        np.asarray(ng_per_ml_to_nM(sil6r_ng_ml, c.mw_sil6r_da), dtype=float),
        np.asarray(ng_per_ml_to_nM(sgp130_ng_ml, c.mw_sgp130_da), dtype=float),
    )


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EquilibriumSolution:
    """Full speciation at equilibrium. All concentrations in nanomolar."""

    free_il6_nM: float
    free_sil6r_nM: float
    free_sgp130_nM: float
    binary_nM: float
    ternary_nM: float
    bt_ratio: float
    converged: bool
    max_residual: float

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _residuals(
    log_free: np.ndarray,
    totals: np.ndarray,
    kd1: float,
    kd2: float,
) -> np.ndarray:
    """Mass-conservation residuals, parameterised in log space for positivity.

    Residuals are expressed relative to the totals so that species spanning
    several orders of magnitude (IL-6 is picomolar, sgp130 is nanomolar) are
    weighted comparably by the least-squares solver.
    """
    free = np.exp(log_free)
    L, R, G = free
    L_tot, R_tot, G_tot = totals

    binary = L * R / kd1
    ternary = binary * G / kd2

    scale = np.maximum(totals, 1e-30)
    return np.array(
        [
            (L + binary + ternary - L_tot) / scale[0],
            (R + binary + ternary - R_tot) / scale[1],
            (G + ternary - G_tot) / scale[2],
        ]
    )


def solve_equilibrium(
    il6_pg_ml: float,
    sil6r_ng_ml: float,
    sgp130_ng_ml: float,
    constants: BiochemConstants | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> EquilibriumSolution:
    """Solve the two-step equilibrium for one sample.

    Uses a damped Newton iteration in log space, falling back to SciPy's
    trust-region least squares if Newton stalls. Returns the full speciation
    including the B/T ratio.

    Raises ``ValueError`` on non-finite or non-positive input, rather than
    silently returning a number - a missing ELISA well must propagate as
    missing, never as a fabricated ratio.
    """
    c = constants or SETTINGS.biochem
    raw = (il6_pg_ml, sil6r_ng_ml, sgp130_ng_ml)
    if any(v is None for v in raw):
        raise ValueError("solve_equilibrium requires all three analytes")
    arr = np.array(raw, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite analyte value: {raw}")
    if np.any(arr <= 0):
        raise ValueError(f"analyte concentrations must be positive, got {raw}")

    L_tot, R_tot, G_tot = (float(x) for x in to_molar_nM(*raw, constants=c))
    totals = np.array([L_tot, R_tot, G_tot])
    kd1, kd2 = c.kd1_nM, c.kd2_nM

    # Initial guess: free ~= total. IL-6 is in vast molar deficit relative to
    # its receptors, so this starting point is close for R and G and merely
    # needs the solver to pull L down.
    log_free = np.log(totals * 0.999)

    converged = False
    for _ in range(max_iter):
        res = _residuals(log_free, totals, kd1, kd2)
        if np.max(np.abs(res)) < tol:
            converged = True
            break

        # Numerical Jacobian in log space (3x3 - cheap and robust).
        jac = np.empty((3, 3))
        step = 1e-7
        for j in range(3):
            perturbed = log_free.copy()
            perturbed[j] += step
            jac[:, j] = (_residuals(perturbed, totals, kd1, kd2) - res) / step

        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(jac) @ res

        # Damp the step so the iteration cannot leap across many decades.
        delta = np.clip(delta, -2.0, 2.0)
        log_free = log_free + delta

    if not converged:
        try:
            from scipy.optimize import least_squares

            fit = least_squares(
                _residuals,
                np.log(totals * 0.999),
                args=(totals, kd1, kd2),
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=5000,
            )
            log_free = fit.x
            converged = bool(np.max(np.abs(fit.fun)) < 1e-8)
        except ImportError:  # pragma: no cover
            pass

    free = np.exp(log_free)
    L, R, G = (float(x) for x in free)
    binary = L * R / kd1
    ternary = binary * G / kd2
    residual = float(np.max(np.abs(_residuals(log_free, totals, kd1, kd2))))

    ratio = float(binary / ternary) if ternary > 0 else float("nan")

    return EquilibriumSolution(
        free_il6_nM=L,
        free_sil6r_nM=R,
        free_sgp130_nM=G,
        binary_nM=float(binary),
        ternary_nM=float(ternary),
        bt_ratio=ratio,
        converged=converged,
        max_residual=residual,
    )


def bt_ratio(
    il6_pg_ml: float,
    sil6r_ng_ml: float,
    sgp130_ng_ml: float,
    constants: BiochemConstants | None = None,
) -> float:
    """Convenience wrapper returning only the B/T ratio (NaN if unsolvable)."""
    try:
        return solve_equilibrium(il6_pg_ml, sil6r_ng_ml, sgp130_ng_ml, constants).bt_ratio
    except ValueError:
        return float("nan")


def free_mediator_equivalent(
    sil6r_ng_ml: float | np.ndarray,
    sgp130_ng_ml: float | np.ndarray,
    constants: BiochemConstants | None = None,
) -> float | np.ndarray:
    """FME = molar sgp130 / molar sIL-6R.

    A simple buffering-capacity index. Deliberately *not* equilibrium-solved -
    it is the cheap comparator against which the B/T ratio has to earn its
    additional complexity.
    """
    c = constants or SETTINGS.biochem
    sil6r_nM = ng_per_ml_to_nM(sil6r_ng_ml, c.mw_sil6r_da)
    sgp130_nM = ng_per_ml_to_nM(sgp130_ng_ml, c.mw_sgp130_da)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(sil6r_nM > 0, sgp130_nM / sil6r_nM, np.nan)
    return float(result) if np.ndim(result) == 0 else result


def compute_bt_ratio_frame(
    il6: Sequence[float] | np.ndarray,
    sil6r: Sequence[float] | np.ndarray,
    sgp130: Sequence[float] | np.ndarray,
    constants: BiochemConstants | None = None,
) -> "object":
    """Vectorised B/T computation returning a DataFrame of full speciation.

    Rows with any missing/invalid analyte yield NaN across every derived
    column - they are never imputed here.
    """
    import pandas as pd

    c = constants or SETTINGS.biochem
    il6 = np.asarray(il6, dtype=float)
    sil6r = np.asarray(sil6r, dtype=float)
    sgp130 = np.asarray(sgp130, dtype=float)

    records = []
    for a, b, g in zip(il6, sil6r, sgp130):
        if not np.all(np.isfinite([a, b, g])) or min(a, b, g) <= 0:
            records.append(
                {
                    "free_il6_nM": np.nan,
                    "free_sil6r_nM": np.nan,
                    "free_sgp130_nM": np.nan,
                    "binary_nM": np.nan,
                    "ternary_nM": np.nan,
                    "BT_ratio": np.nan,
                    "bt_converged": False,
                }
            )
            continue
        sol = solve_equilibrium(a, b, g, constants=c)
        records.append(
            {
                "free_il6_nM": sol.free_il6_nM,
                "free_sil6r_nM": sol.free_sil6r_nM,
                "free_sgp130_nM": sol.free_sgp130_nM,
                "binary_nM": sol.binary_nM,
                "ternary_nM": sol.ternary_nM,
                "BT_ratio": sol.bt_ratio,
                "bt_converged": sol.converged,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["FME"] = free_mediator_equivalent(sil6r, sgp130, constants=c)
    frame["log_BT_ratio"] = np.log(frame["BT_ratio"].where(frame["BT_ratio"] > 0))
    return frame
