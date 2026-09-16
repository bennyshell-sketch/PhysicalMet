# PORAF B/T Ratio Prediction Pipeline

Analysis pipeline and interface for the PORAF study (NCSEM, Loughborough),
investigating whether the IL-6 trans-signalling **B/T ratio** predicts physical
function decline and trajectory in older adults.

The ELISA data does not exist yet. Everything here is built and validated
against synthetic data that matches the real workbook schema, so that when the
real file arrives you change **one configuration value** and nothing else.

---

## Quick start

```bash
pip install -r requirements.txt

python -m poraf_bt.generate_synthetic_data   # create the synthetic workbook
python -m poraf_bt.validation                # 17 correctness checks
streamlit run app/app.py                     # open the interface
```

---

## Pointing it at the real data

This is the only change required.

1. Put the real workbook in `data/real/` as
   `PORAF_BT_ratio_candidate_selection_n100.xlsx`.
2. Set the data source, by any one of:

   ```bash
   export PORAF_DATA_SOURCE=real          # environment variable
   ```
   ```yaml
   # settings.local.yaml (gitignored)
   data_source: real
   ```
   or edit the `data_source` default in `poraf_bt/config.py`.

3. Run as normal.

If the real file uses different column or sheet names, add them to
`COLUMN_ALIASES` / `SHEET_ALIASES` in `poraf_bt/data_pipeline.py`. No analysis
code changes. Filenames themselves are configurable under `PathConfig`.

**Before trusting the first real-data run**, check the *Cleaning actions* tab
on the Data screen. It lists every correction the pipeline applied. If the real
workbook has a quirk that isn't handled, it shows up there as a surprise.

---

## What the pipeline does

| Module | Responsibility |
|---|---|
| `poraf_bt/biochem.py` | Two-step equilibrium solved numerically for the B/T ratio |
| `poraf_bt/generate_synthetic_data.py` | Synthetic workbook + ground-truth sidecar |
| `poraf_bt/data_pipeline.py` | Loading, cleaning, reshaping, quality reporting |
| `poraf_bt/group_comparison.py` | Kruskal-Wallis + sex-adjusted ANCOVA |
| `poraf_bt/decline_model.py` | Primary logistic model; gradient boosting as sensitivity only |
| `poraf_bt/trajectory_model.py` | Mixed-effects trajectory + individual prediction intervals |
| `poraf_bt/guardrails.py` | Abstention: extrapolation and interval-width checks |
| `poraf_bt/recommend.py` | Evidence-matched recommendations from a curated library |
| `poraf_bt/learning/` | Multi-source ingestion, model registry, gated retraining |
| `poraf_bt/validation.py` | The correctness harness |
| `app/` | Streamlit interface |

### The B/T ratio

Solved, not approximated:

```
IL-6 + sIL-6R          <-> [IL-6.sIL-6R]            Kd1 = 10 nM
[IL-6.sIL-6R] + sgp130 <-> [IL-6.sIL-6R.sgp130]     Kd2 =  1 nM
```

Assay units (pg/mL, ng/mL) are converted to molarity first, then free
concentrations are solved by damped Newton iteration in log space, with a
SciPy least-squares fallback. Mass balance holds to ~1e-13. The ratio collapses
analytically to `Kd2 / [free sgp130]`, which is used as an internal consistency
check — never as a shortcut past the solve.

`FME` (molar sgp130 / molar sIL-6R) is the cheap comparator the B/T ratio has
to beat to justify its extra complexity.

---

## How it avoids making things up

This was the primary design constraint, and it costs the system the ability to
always produce a number. That is the intended trade.

**Models abstain rather than guess.** A prediction is released only if:

- the model beat chance on its own permutation test;
- the person's covariates fall inside the region the model was trained on
  (per-variable range *and* multivariate Mahalanobis distance — a person can be
  unremarkable on every single variable and still be a combination nobody in
  the sample resembled);
- the honest uncertainty is narrow enough to carry information.

Otherwise the interface states that no prediction was issued and why. On the
current synthetic data the decline model fails its permutation test, so it
refuses to give individual risks. That is the correct behaviour.

**The permutation test drives the verdict, not the AUC.** With n=100 and three
predictors, a cross-validated AUC of 0.60 is comfortably reachable from noise.
Every AUC is reported next to the proportion of label permutations that matched
it.

**Case-control sampling is corrected, and said out loud.** The substudy is 50
decliners and 50 stable by design. A King & Zeng prior correction maps the
intercept onto the observed cohort prevalence, and any uncorrected probability
is labelled as describing the 50/50 sample rather than the population.

**Forecasts use prediction intervals, not confidence intervals.** A confidence
interval covers the group mean; a forecast for one person needs the between-
participant and residual variance too. Held-out coverage of the nominal 95%
interval measures 95.2%.

**Recommendations are matched, not generated.** `poraf_bt/knowledge/interventions.yaml`
is the only source. An entry appears only when a real measurement fires an
explicit trigger, and carries its evidence grade (A–D) and citation. Where the
evidence does not exist, the library says so: no intervention has been shown to
shift the B/T ratio, and that is displayed as an evidence gap rather than
quietly omitted or softened into advice.

---

## Training it on new data

The `Training and data` screen, or the API:

```python
from poraf_bt.learning import DataSource, ingest_file, retrain_and_evaluate

source = DataSource(source_id="partner_a", label="Partner Cohort A",
                    kind="cohort", trust=0.8)
result = ingest_file("partner_data.csv", source)
print(result.summary())      # accepted / quarantined / unmapped columns
```

Incoming columns are matched against the canonical schema by alias. Rows with
implausible values are **quarantined with a stated reason**, never coerced into
range — a grip strength of 250 kg is a unit error, and clipping it to 90 would
teach the model something false. Every batch is stored with a manifest
recording origin, checksum and what was dropped.

Retraining fits a challenger and promotes it only if **all three** hold:

1. it beats the champion on a participant-disjoint holdout by a real margin;
2. the margin survives a paired bootstrap;
3. it passes its own permutation test.

Condition 3 is the one that matters. Without it, a model that has learned
nothing can still be promoted for scoring marginally better noise than its
predecessor.

Distribution shift between the reference cohort and incoming data is reported
(KS test + standardised mean difference) but never blocks ingestion — a new
cohort may legitimately differ, and the point is that you are told.

---

## Validation

```bash
python -m poraf_bt.validation     # 17 checks
python -m pytest tests/ -q        # 56 tests
```

Four families of check:

- **Leakage** — participant-aware splits, plus a demonstration that a naive row
  split really would leak (so the guard isn't passing vacuously).
- **Null recovery** — null data in, null result out, and abstention downstream.
- **Effect recovery** — a *known simulated effect* is detected and its variance
  components recovered. A pipeline that only ever returns "no effect" is broken,
  not conservative; this is the check that tells the two apart.
- **Calibration** — empirical coverage of the nominal 95% prediction interval.

---

## Known limitations

- **Random-slope variance is underestimated.** With 2–4 visits over 12 months
  and ~1.9 kg of measurement noise, a per-participant slope of 0.07 kg/month
  produces less signal than the noise it sits in. REML shrinks it (recovering
  ~0.014 against a simulated 0.07). This is correct behaviour for the data
  available, not a fixable bug, and it means individual slopes are pulled
  toward the cohort mean. More timepoints would be the only real remedy.
- **The synthetic substudy's sex split is emergent.** The spec quotes roughly
  36F/14M and 34F/16M; the generator produces about 29F/21M and 30F/20M because
  case selection follows simulated grip decline rather than being imposed. The
  mechanism is honest; the exact split is not matched.
- **The `older_age_threshold` differs between sources.** The build spec says
  65y; the confirmed PORAF protocol says 60y (Group 0, n=281). The config
  defaults to 60 and the value is configurable.
- **Sample size does not support deep learning**, and nothing here attempts it.
  Mixed models and logistic regression are primary; gradient boosting is a
  labelled sensitivity check only.

---

## Prior result worth keeping in view

An earlier RFU-based ML analysis of this same pathway found no signal
(RF CV-AUC 0.42, permutation p=0.75). On the default synthetic data this
pipeline reproduces that shape — AUC 0.41, permutation p=0.93 — because no
effect was simulated. Any apparent signal on synthetic data is a pipeline
correctness check, never evidence about the biology.
