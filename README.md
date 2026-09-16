# Dive

A clinical research platform for predicting physical function decline and
trajectory in older adults, from longitudinal cohort data.

Dive ingests physical function measures, body composition, imaging-derived
measurements (DEXA bone density, CT body composition), inflammatory biomarkers
and anything else you register; fits inferential and predictive models with
honest uncertainty; and improves as data accumulates — under gates that stop it
improving itself into noise.

It is built and validated against synthetic data that matches a real cohort
schema, so a deployment changes **one configuration value** to point at real
data and nothing else.

---

## Quick start

```bash
pip install -r requirements.txt

python -m dive.generate_synthetic_data   # create the synthetic workbook
python -m dive.validation                # 17 correctness checks
streamlit run app/app.py                     # open the interface
```

---

## Design commitments

Three things shape every decision in this codebase.

**It abstains rather than guessing.** A model that cannot beat chance on its own
permutation test does not issue individual predictions. A profile outside the
covariate region the model was trained on gets a refusal with reasons, not a
number. On the bundled synthetic data the decline model *does* refuse, and that
is the correct behaviour.

**It cannot cite something that does not exist.** The advisor is handed
retrieved studies and its citations are intersected with that set afterwards.
A model-supplied identifier that was not retrieved is discarded.

**It does not send identifiers anywhere.** Everything bound for an external
model passes one guard, which de-identifies, re-scans, and blocks on a hit.

---

## Pointing it at the real data

This is the only change required.

1. Put the real workbook in `data/real/` as `cohort_biomarkers.xlsx`
   (filenames are configurable under `PathConfig`).
2. Set the data source, by any one of:

   ```bash
   export DIVE_DATA_SOURCE=real           # environment variable
   ```
   ```yaml
   # settings.local.yaml (gitignored)
   data_source: real
   ```
   or edit the `data_source` default in `dive/config.py`.

3. Run as normal.

If the real file uses different column or sheet names, add them to
`COLUMN_ALIASES` / `SHEET_ALIASES` in `dive/data_pipeline.py`. No analysis
code changes. Filenames themselves are configurable under `PathConfig`.

**Before trusting the first real-data run**, check the *Cleaning actions* tab
on the Data screen. It lists every correction the pipeline applied. If the real
workbook has a quirk that isn't handled, it shows up there as a surprise.

---

## What the pipeline does

| Module | Responsibility |
|---|---|
| `dive/biochem.py` | Two-step equilibrium solved numerically for the B/T ratio |
| `dive/metrics.py` | Registry of health metrics Dive can ingest and test |
| `dive/generate_synthetic_data.py` | Synthetic workbook + ground-truth sidecar |
| `dive/data_pipeline.py` | Loading, cleaning, reshaping, quality reporting |
| `dive/group_comparison.py` | Kruskal-Wallis + sex-adjusted ANCOVA |
| `dive/decline_model.py` | Primary logistic model; gradient boosting as sensitivity only |
| `dive/trajectory_model.py` | Mixed-effects trajectory + individual prediction intervals |
| `dive/guardrails.py` | Abstention: extrapolation and interval-width checks |
| `dive/recommend.py` | Evidence-matched recommendations from a curated library |
| `dive/literature.py` | Retrieval of real, verifiable studies + citation verification |
| `dive/advisor.py` | Participant-specific advice grounded in retrieved evidence |
| `dive/calibration.py` | Scanner QC, precision, least significant change, comparability |
| `dive/fairness.py` | Subgroup auditing; protected attributes screened out of models |
| `dive/evidence_grading.py` | OCEBM, GRADE, Cochrane RoB 2, AMSTAR-2 |
| `dive/privacy.py` | The egress boundary: de-identification, scanning, audit |
| `dive/security.py` | Audit chain, encryption at rest, auth, upload validation |
| `dive/providers.py` | Claude / Gemini / local models behind one guarded interface |
| `dive/learning/` | Ingestion, registry, gated retraining, feature search, autolearn |
| `dive/validation.py` | The correctness harness |
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

**Recommendations are matched, not generated.** `dive/knowledge/interventions.yaml`
is the only source. An entry appears only when a real measurement fires an
explicit trigger, and carries its evidence grade (A–D) and citation. Where the
evidence does not exist, the library says so: no intervention has been shown to
shift the B/T ratio, and that is displayed as an evidence gap rather than
quietly omitted or softened into advice.

---

## Adding any health metric, including imaging

Dive ships with 137 core metrics across fifteen domains — physical function,
body composition, imaging, biochemical, microbiological, histological, gross
anatomical, genetic, epigenetic, geographic context, cognition and activity.
See [docs/OMICS.md](docs/OMICS.md) for what each domain contains and where the
measurement traps are.

Older material, still accurate:
**DEXA** (T-score and Z-score, site-specific BMD, trabecular bone score,
appendicular lean mass), **CT** (L3 skeletal muscle area, muscle attenuation in
Hounsfield units, visceral / subcutaneous / intermuscular adipose area,
vertebral trabecular attenuation, aortic calcification), routine bloods,
frailty, cognition and activity.

A boundary worth stating: Dive consumes **derived measurements** — the numbers a
radiologist, a DEXA report or a segmentation tool produces — not raw DICOM pixel
data. Learning directly from images needs sample sizes in the thousands and a
segmentation pipeline that is itself regulated software. Consuming validated
derived measures gets most of the signal at a fraction of the risk, and keeps
every predictor something a clinician can interrogate.

It is not limited to what it ships with. Register a metric and it
becomes ingestable immediately, with its plausible range enforced:

```python
from dive.metrics import HealthMetric, MetricRegistry

MetricRegistry().register(HealthMetric(
    name="serum_vitamin_d", label="Serum 25-OH vitamin D", unit="nmol/L",
    domain="clinical", minimum=5, maximum=300, higher_is_better=True,
    aliases=("vit_d", "vitamin_d_25oh"),
))
```

Incoming files using any registered alias now map automatically, and values
outside the plausible range are quarantined as unit errors rather than learned.

**Registering a metric does not put it into a model.** It makes the metric
eligible. Whether it earns a place is decided on held-out participants:

```python
from dive.learning import search_features
result = search_features(data.substudy, n_permutations=200)
print(result.verdict)
```

This is the part that is easy to get wrong. Try twenty metrics, keep whichever
raises cross-validated AUC the most, and you will find an improvement almost
every time — because the maximum of twenty noisy estimates is biased upward
even when all twenty are pure noise. On Dive's own synthetic data, a metric
drawn from `np.random.normal` raised AUC by **+0.098**, which would clear any
plain threshold.

So the null is built from the *whole search*: the outcome labels are shuffled
and the entire selection — including "pick the best candidate" — is re-run.
That answers the question that actually matters, and it rejects the noise
metric (p = 0.31) while adopting a genuinely informative one (+0.401,
p = 0.012).

A metric is adopted only if it clears a minimum effect size, has enough
coverage to be evaluated, and beats that selection-level null.

---

## Privacy: what leaves, and what cannot

Start from the honest position: **if Dive sends text to Claude or Gemini, that
provider processes it.** No header, setting or contract makes a provider unable
to see what you send. So the only enforceable guarantee is about what is sent,
and that is what `dive/privacy.py` controls.

Every outbound call — from any module, to any provider — passes through one
`EgressGuard`. There is no other path to an external model, which is what makes
the audit log a complete record rather than a sample.

| Mode | Behaviour |
|---|---|
| `none` | Nothing participant-derived leaves. Maximum privacy, least specificity. |
| `deidentified` *(default)* | Clinical values only: no identifiers, no dates, no free text, age in 5-year bands, values rounded. |
| `full` | **Refused by the guard.** The mode exists so the configuration is explicit; there is no legitimate reason to send raw records to a general-purpose LLM. |

Enforcement is fail-closed throughout. A field not on the allowlist is dropped,
not passed through as probably-harmless. A payload still matching an identifier
pattern *after* de-identification is blocked outright rather than sent with a
warning — silently rewriting it would hide that something upstream leaked. If
the scanner itself raises, the egress is blocked.

On a real record from the bundled cohort: **8 fields sent, 57 withheld.** The
Privacy screen shows this for any participant, and the audit log records a hash
of what left, never the payload.

```yaml
privacy:
  egress_mode: deidentified   # none | deidentified
  offline_only: false         # true is an absolute kill switch
  min_aggregate_n: 5          # small-cell suppression
```

**The zero-egress option.** Point `DIVE_LOCAL_LLM_URL` at an OpenAI-compatible
endpoint on your own hardware (llama.cpp, vLLM, Ollama). Nothing leaves your
network at all, and the guard still runs in case a "local" URL is misconfigured
to a public one.

**What is honestly not claimed:** de-identified is not anonymous. An unusual
combination of measurements, cross-referenced against another dataset, remains
a remote re-identification route. `egress_mode: none` and a local model are the
only ways to reduce that to zero, and both are supported.

---

## Security

These controls raise the cost of an attack and make a successful one visible
afterwards. **They do not make Dive unbreakable — no software is**, and anyone
selling you otherwise is selling you something else.

| Control | Implementation |
|---|---|
| Tamper-evident audit | Hash-chained log; a deleted or edited entry breaks the chain and `verify()` finds it |
| Encryption at rest | AES-GCM via `DIVE_ENCRYPTION_KEY`; refuses to fall back to a hand-rolled cipher unless explicitly allowed |
| Password storage | scrypt, per-password salt, 12-character minimum, account lockout |
| Sessions | HMAC-signed, expiring, constant-time verification — a client-edited role fails |
| Upload validation | Size cap, extension allowlist, magic-byte check, path-traversal rejection |
| Formula injection | Leading `=`, `+`, `-`, `@` neutralised on export |
| Secret handling | Keys held by reference; `repr()` and logs show a fingerprint, never the value |
| Identifier leakage | Audit entries are redacted before they are written |

Role capabilities: `viewer` sees aggregate results; `analyst` sees individual
records, runs models, ingests data; `admin` additionally manages users,
promotes models, changes the privacy policy and exports data.

**What remains the deployer's job** and cannot be solved in application code:
TLS and certificate management, network isolation, OS and dependency patching,
key custody and rotation, encrypted backups with tested restores, and an
independent penetration test before real patient data is loaded.

**Two governance points worth settling early**, because they shape the product
rather than the code. UK GDPR / DPA 2018 applies to this data, and a DPIA needs
to cover the model providers specifically. And if Dive's output is used to guide
care rather than research, it is very likely a medical device under UKCA / MHRA
and EU MDR — a classification question that is far cheaper to answer now than
after a sale.

---

## The evidence advisor

The curated library in `dive/recommend.py` fires on fixed thresholds and
returns the same wording for anyone who crosses them. The advisor
(`dive/advisor.py`) is the participant-specific layer: it reasons about *this*
combination of values and says where the trial populations differ from the
person in front of it.

It uses the Claude API, which makes fabricated citations the obvious risk. A
model asked to cite literature produces references that look correct in every
respect — plausible authors, a real journal, a well-formed DOI — and do not
exist. Three things make that structurally impossible here:

1. **The model never supplies a citation.** Studies are retrieved first from
   Europe PMC (falling back to PubMed), restricted to systematic reviews,
   meta-analyses, trials and cohort studies with abstracts.
2. **Only retrieved records are passed in**, and the model is instructed it may
   cite nothing else.
3. **Every identifier it returns is checked against the retrieved set.**
   Anything else is discarded, and a recommendation left with no verifiable
   source is dropped entirely. The interface reports how many were discarded.

The model also never supplies a fact about the participant — measurements come
from the pipeline as a fixed block — and it is instructed not to forecast, because
the decline and trajectory models own prediction, including the decision to
abstain.

Requires a provider key and network access to Europe PMC or PubMed. Without
either, the advisor reports itself unavailable and Dive falls back to the
curated library. It will not write advice from recall.

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Claude
export GEMINI_API_KEY=AIza...           # Gemini
export DIVE_LOCAL_LLM_URL=http://localhost:8000/v1   # or neither, and stay local
```

The router tries providers in order and re-runs the privacy guard for each — a
payload cleared for one destination is not thereby cleared for another.

---

## Learning on its own

`dive/learning/autolearn.py` runs the whole cycle unattended: ingest what has
arrived, check the new data resembles the old, test whether any metric now earns
a place, retrain, promote only what is demonstrably better.

The word doing the work is *only*. An unsupervised loop that retrains on
whatever arrives and adopts whatever scores highest does not get better over
time — it drifts, confidently, and the drift is invisible because the same loop
that caused it also reports on it. So every stage can decline to act, and
declining is the default:

| Stage | Can refuse when |
|---|---|
| Ingest | too few new rows; implausible values are quarantined, never clipped |
| Drift | the new data looks like a different population — the cycle **halts** rather than fitting through it |
| Feature search | the gain does not beat a permutation null built from the whole search |
| Retrain | the challenger does not beat the incumbent on held-out participants, or fails a bootstrap |
| Promote | refused outright for a model that cannot beat chance |

Plus a rate limit between runs, and an automatic pause after three consecutive
failures — repeatedly retrying a broken pipeline is how a loop quietly degrades
a working model.

**A cycle that changes nothing is a successful cycle.** The failure mode this
is built around is a system that always finds something.

```python
from dive.learning import AutoLearner
result = AutoLearner().run_cycle()
print(result.summary())
```

---

## Training it on new data

The `Training and data` screen, or the API:

```python
from dive.learning import DataSource, ingest_file, retrain_and_evaluate

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
python -m dive.validation     # 17 checks
python -m pytest tests/ -q        # 181 tests
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
- **The advisor's literature retrieval and both provider calls are untested
  against the live services.**
  The environment Dive was built in blocks Europe PMC and PubMed (403) and had
  no provider key, so retrieval, query construction and the Claude and Gemini
  calls have not been exercised end-to-end. The parts that matter — citation
  verification and the privacy guard — are tested offline against fabricated
  identifiers and real identifier patterns. Run the advisor once with network
  access and a key before relying on it.
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
