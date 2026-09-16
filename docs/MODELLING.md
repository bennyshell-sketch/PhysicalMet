# Modelling decisions

Notes on choices that are not obvious from the code, and the reasoning behind
them. Written for whoever picks this up next, including future-you.

## Why mixed-model time is carried in years

Visits are recorded in months, and the first implementation used months as the
time variable. The fitted random-slope SD came back at **5.5 kg/month** against
a simulated truth of 0.07, and the resulting 12-month prediction interval
spanned −110 to +156 kg.

The cause is variance scaling. Grip has a between-participant SD of ~7 kg, so
the random-intercept variance is ~49. A slope SD of 0.07 kg/month gives a slope
variance of ~0.005. That is a ratio of roughly 10,000:1 in the same covariance
matrix, and the REML optimiser does not identify it — lbfgs, cg and powell all
returned different answers, and the Hessian was not positive definite.

Rescaling time to years multiplies the slope variance by 144, bringing the
ratio to about 70:1. All three optimisers then agree, the Hessian is well
behaved, and the recovered components sit close to the truth.

`MONTHS_PER_YEAR` in `trajectory_model.py` handles the conversion. Fixed-effect
slopes are reported per year. `_fit_with_fallback` still tries optimisers in
order, because the real data may be less well conditioned than the synthetic.

## statsmodels `cov_re` is already in outcome units

An early version multiplied `fit.cov_re` by `fit.scale`, on the assumption that
statsmodels reports the random-effects covariance scaled by residual variance.
It does not. The multiplication inflated every variance component — and
therefore every prediction interval — by the residual variance.

Verified empirically against a simulation with known components before the fix
was applied. If you are ever unsure which convention a library uses, simulate
data with a known answer and check; the docstring is not always decisive.

## Why every predictor is centred, including sex

Originally only the continuous predictors were centred. With sex left at 0/1,
the `time` coefficient was the slope for sex=0 (males), not the population
average — but the verdict string described it as the cohort average. On the
synthetic data that was −0.48 kg/year reported against a true within-
participant slope of −0.67.

Centring sex makes the `time` coefficient the population-average slope, which
is what the verdict claims. The sex coefficient still reads as the male-female
difference.

## Which slope is "the truth"

Validating slope recovery took three attempts, because there are three
different quantities and only one is the estimand:

| Quantity | Synthetic value | What it is |
|---|---|---|
| `grip_slope_mean` parameter | −0.54 kg/year | The per-participant linear slope only |
| Pooled OLS slope | −3.11 kg/year | Confounded by between-participant differences and by who attends which visit |
| **Within-participant slope** | **−0.67 kg/year** | What a mixed model with a random intercept targets |

The bare parameter is wrong as a target because the generator also applies a
post-fall dip and a non-recovery decrement for the Declined Function group.
Those are real features of the simulated cohort and the population-average
trajectory legitimately includes them. The pooled OLS slope is wrong for the
usual reason pooled estimates are wrong with repeated measures.

The generator now records `realised.within_slope_per_year` in
`ground_truth.json`, and validation compares against that. The model recovers
−0.71 against −0.68.

## Why the permutation test uses a lighter CV scheme

The headline AUC uses repeated stratified k-fold (10 folds x 20 repeats). Running
that inside 500 permutations would be 100,000 model fits.

The permutation test instead uses a single 5-fold pass, and — importantly —
recomputes the **observed** AUC under the same light scheme, so the comparison
is like-for-like. Reporting a repeated-CV AUC against a null distribution built
from single-pass CV would bias the p-value.

## Why gradient boosting is not the headline

It is included because the spec asks for a robustness check, and it answers a
real question: does a flexible model find structure the logistic regression
missed? With n=100 its cross-validated AUC is unstable across seeds, and it
cannot support the inferential claims the thesis chapter needs. The module
labels it `EXPLORATORY ROBUSTNESS CHECK - NOT THE PRIMARY RESULT` and the
interface never promotes it.

## Why the subgroup palette changed

The PORAF plotting convention specifies `#2166ac` / `#4dac26` / `#d95f02` /
`#d62728` for the four faller subgroups. Those colours do not separate:

- Recovered Function vs Declined Function: ΔE 9.7 in **normal** vision
- Maintained vs Recovered: ΔE 2.8 under deuteranopia

RF and DF are the two clinically opposite outcomes in the study. In a printed
thesis figure, a substantial fraction of readers could not tell them apart, and
under deuteranopia neither could most readers distinguish MF from RF.

The interface uses Okabe-Ito instead (`#0072B2` / `#009E73` / `#E69F00` /
`#D55E00`), which is published as colour-vision-deficiency safe and clears
every separation check with the worst adjacent pair at ΔE 11.4. The semantic
mapping (blue = reference, green = maintained, amber = recovered, vermillion =
declined) is preserved.

If the old palette must be retained for consistency with existing figures, the
subgroups need a second encoding — direct labels, markers or texture — so
identity never rests on colour alone.

## Abstention is a feature, not a failure mode

The system currently refuses to give individual decline risks on synthetic
data, because the model does not beat chance. This looks like a broken feature
and is the opposite.

The alternative is a system that produces a confident-looking percentage from a
model with AUC 0.41. That number would be indistinguishable in presentation
from one produced by a model that works, and it would be acted on. Given that
the prior RFU analysis on this same pathway also found nothing, the realistic
case is that the real data behaves similarly — so the abstention path is the
one most likely to matter in practice, and it is tested accordingly.
