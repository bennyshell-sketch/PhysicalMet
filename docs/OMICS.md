# Measurement domains

Dive ships with 137 core metrics across fifteen domains. This document covers
the ones where the measurement is easy to record and easy to misuse.

The governing rule: **Dive does not assert what any of these predict.** A
metric being present in the registry is not a claim that it matters. Whether a
variable carries signal in *your* cohort is settled by
`dive/learning/feature_search.py` against a permutation null built from the
whole search — not by a literature claim, and never by a language model.

That distinction is the whole anti-hallucination design. It would have been
easy to attach "ACTN3 R577X is associated with power performance" to a metric
and let it colour the output. Dive does not, because an association reported in
one cohort is a hypothesis about yours.

---

## Biochemical

24 metrics: bone turnover (P1NP, CTX, bone ALP, osteocalcin, PTH, sclerostin),
endocrine (IGF-1, testosterone, oestradiol, SHBG, cortisol, TSH), metabolic
(HbA1c, fasting glucose, HOMA-IR), renal (eGFR, cystatin C, creatinine), and
nutritional and muscle markers.

Two things make these comparable or not:

**Collection conditions.** CTX is the clearest case — it has a marked circadian
rhythm and is suppressed by feeding, so a non-fasting afternoon sample is not
comparable with a fasting morning one. Record the conditions or the number is
not interpretable.

**Assay and units.** HbA1c is stored in IFCC mmol/mol, not DCCT %. Bone turnover
markers differ between platforms. Dive stores what you give it and enforces a
plausibility range; it does not convert between assays, because conversion
factors are assay-pair-specific and inventing one would be exactly the failure
this system is built to avoid.

**eGFR specifically:** use the race-free CKD-EPI 2021 equation. See
[fairness](#ethnicity-race-and-geography).

---

## Microbiological

Alpha diversity (Shannon, Simpson, Chao1, observed ASVs), the
Firmicutes:Bacteroidetes ratio, faecal short-chain fatty acids and calprotectin,
and LPS-binding protein.

**Diversity indices are pipeline-dependent.** A Shannon index depends on the 16S
region sequenced, the rarefaction depth, the reference database, and whether
ASVs or OTUs were called. Two cohorts processed differently produce
non-comparable numbers that look perfectly comparable. Record the pipeline as
source metadata, and treat a change of pipeline the way you would treat a change
of scanner.

The F:B ratio is included because it is widely reported, not because it is
well-founded; it is a coarse summary of two very large phyla and its
associations replicate poorly. It is a candidate feature like any other, and the
feature search is free to reject it.

---

## Histological

Muscle biopsy morphometry (fibre type proportions, type II fibre CSA,
capillary-to-fibre ratio, satellite cells per fibre) and bone histomorphometry
in standard ASBMR nomenclature (BV/TV, Tb.Th, Tb.N, Tb.Sp, MAR, BFR/BS, OS/BS).

Histology is the smallest-n data most cohorts will ever have, because it needs a
biopsy. That has a direct modelling consequence: a histological variable will
usually fail the feature search's coverage floor, and *that is the correct
outcome*. A metric measured on 20 of 300 participants cannot be evaluated
honestly, and the search says so rather than fitting it on the subset.

---

## Gross anatomical

Segment lengths, circumferences, hip axis length, kyphosis angle, joint range of
motion.

Tibia and forearm length are not incidental — pQCT sites are defined as a
percentage of limb length, so the length determines *where* the scan was taken.
A cohort with inconsistent limb-length measurement has inconsistent scan sites,
and the density values move accordingly.

---

## Genetic

Individual variants are stored as effect-allele dosage (0, 1 or 2); polygenic
scores are stored standardised.

**Dive makes no claim about any variant.** The registry records that
`actn3_r577x_dosage` exists and is bounded 0–2. Whether it predicts anything
here is an empirical question this cohort answers.

**Polygenic scores transfer poorly across ancestries.** Most published scores
were derived in European-ancestry samples and lose a large share of their
predictive accuracy elsewhere. Applied across a diverse cohort, a PRS can look
predictive overall while being near-useless in some groups — and the aggregate
AUC hides exactly that. Run the fairness audit stratified by ancestry before
trusting one.

**A rare genotype is identifying on its own.** Genotypes and ancestry components
are therefore in the privacy module's identifying-keys list and never leave the
process.

---

## Epigenetic

Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE, telomere length, and age
acceleration.

**Age acceleration must be computed within your cohort.** It is the residual of
predicted age on chronological age. A vendor report calibrated on a different
population, array platform and normalisation pipeline gives a residual relative
to *their* cohort, which is not the quantity you want.

**Clocks are not comparable across array platforms.** 450K, EPIC and EPICv2
differ in probe content; a clock run on one is not interchangeable with the same
clock run on another. The platform is recorded as a metric for that reason.

---

## Imaging: pQCT and bone marrow adiposity

pQCT contributes volumetric density (total, trabecular, cortical), geometry
(cortical thickness and area, periosteal and endosteal circumference) and
strength (stress-strain index, bone strength index), plus muscle and fat
cross-sectional area and muscle density.

Marrow adiposity is stored separately by method: MR spectroscopy PDFF (the
reference method), water-fat MRI fat fraction, segmented volume, and CT
attenuation.

**These four are not interchangeable and Dive refuses to pool them.** CT
attenuation moves in the *opposite* direction to a fat fraction. They are also
site-specific — lumbar spine, femur and tibia values are not comparable with
each other. `dive/calibration.py::assert_bmat_comparable` blocks the mistake.

**Calibration is not optional.** `dive/calibration.py` holds phantom QC with
drift detection, precision from repeat scans, and the least significant change
(`2.77 × RMS-SD`, the ISCD formula) — the smallest difference that is a real
change rather than machine noise. On realistic pQCT precision the LSC is around
10 mg/cm³, which means a follow-up trabecular density 6 mg/cm³ lower has not
changed. Without this, a multi-site model learns the scanner instead of the
patient.

---

## Ethnicity, race and geography

This is the part with a track record, so Dive treats it differently from every
other domain.

Clinical algorithms that used race as a biological input have had to be
withdrawn or rebuilt, and the pattern is consistent: a social category was
treated as a biological one, and the people it disadvantaged were already
disadvantaged.

| Algorithm | What it did | Outcome |
|---|---|---|
| CKD-EPI 2009 eGFR | Race coefficient raised estimated GFR for Black patients | Later referral and transplant listing; race-free equation recommended 2021 |
| Spirometry "race correction" | Normalised lower measured values as expected | Masked disease; ATS moved to race-neutral equations 2023 |
| MFMU VBAC calculator | Lowered predicted success for Black and Hispanic women | Race and ethnicity removed in the 2021 revision |
| Pulse oximetry | Overestimates saturation in darker skin | Missed hypoxaemia — a device bias, not an algorithm one |

So Dive draws three lines.

**Race and ethnicity are not biology.** They are self-reported social categories
whose meaning depends on the classification system and the society doing the
classifying. `dive/fairness.py::screen_features` refuses them as model features
before anything is fitted. Overriding requires `allow_protected_features=True`
and is logged.

**Genetic ancestry is a different variable and is allowed.** Continuous ancestry
principal components are measurable and not a social category. They are ordinary
candidate features.

**Geography and deprivation are context, not cause.** Their legitimate uses are
reference-range selection, assay standardisation and fairness auditing — never
as though a postcode caused frailty. Where they are used as predictors they must
be labelled as social-determinant proxies.

### What they are for: auditing

A model can be accurate overall and much worse for one group. The aggregate
number hides it; the only way to know is to measure per group.
`audit_subgroups` does that, reporting discrimination and calibration within
each subgroup with small-cell suppression so the audit does not itself disclose
individuals.

When a subgroup is suppressed, the report says so explicitly — absence from the
table is *not* evidence the model works for that group, it is evidence that you
cannot tell.

### Reference populations

This is where geography and ancestry legitimately belong. A T-score is a
standard deviation score against a young-adult reference population, so the
reference database determines the number — change the reference and you change
who is diagnosed, without any patient changing.
`reference_population_note()` records the relevant caveat for each derived
score.

---

## Adding your own

Anything not listed is registered at runtime and becomes ingestable immediately:

```python
from dive.metrics import HealthMetric, MetricRegistry

MetricRegistry().register(HealthMetric(
    name="serum_myostatin", label="Serum myostatin", unit="ng/mL",
    domain="biochemical", minimum=0.1, maximum=200, higher_is_better=False,
    aliases=("myostatin", "gdf8"),
    source="R&D Systems Quantikine ELISA, fasting morning sample",
))
```

The plausible range is the unit-error detector, not a preference about what the
data should look like. Registering makes a metric *eligible*; the feature search
decides whether it earns a place.
