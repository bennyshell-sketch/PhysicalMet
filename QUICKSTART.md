# Using Dive

## Get it running

You need Python 3.10 or newer. Everything else installs itself.

```bash
git clone --branch claude/exciting-ride-s5j9tk https://github.com/bennyshell-sketch/PhysicalMet.git dive
cd dive
./start-dive-mac-linux.sh
```

The repository is private, so cloning will ask for your GitHub credentials.

**On Windows**, double-click **`START-DIVE-windows.bat`**. Step by step,
including installing Python: [WINDOWS.md](WINDOWS.md)

First run takes a few minutes: it builds a local environment, installs
dependencies and generates a synthetic cohort. After that it starts in seconds.
Your browser opens at **http://localhost:8501**. Ctrl+C in the terminal stops it.

No API keys. No internet. Nothing is sent anywhere. It works out of the box on
synthetic data.

To check it installed correctly:

```bash
./start-dive-mac-linux.sh check      # 17 validation checks, then 181 tests
```

---

## What you are looking at

Seven screens in the left sidebar.

**Data** — the cohort, and every correction the pipeline applied. Start here.
The *Cleaning actions* tab is the one to read: it lists what was fixed and why.
On real data, anything surprising in that list is worth chasing.

**Group comparison** — does a measure differ between faller subgroups?
Kruskal-Wallis, then the same question adjusted for sex. Plain answer first,
statistics underneath.

**Individual prediction** — enter a participant's values, get a decline risk and
a 12-month forecast.

**Trajectory** — pick a participant, see their measurements and the projected
path with a prediction interval.

**Recommendations** — interventions whose evidence applies to this person's
measured values, each triggered by a specific measurement and carrying its
grade and citation.

**Training and data** — add new data, register new metrics, calibrate scanners,
test whether a metric earns a place, retrain.

**Privacy and security** — what leaves the process, what is withheld, the audit
log, and the fairness audit.

**Owner console** — locked behind your own key. The full history, what the model
has learned, and source editing. See below.

---

## The first thing that will surprise you

Go to **Individual prediction**, enter anything, press Predict.

The decline risk will say **"No prediction issued"** and explain that the model
does not beat chance.

That is correct, and it is the single most important behaviour in the system. On
the synthetic data there is no signal to find, so the model refuses rather than
producing a confident-looking percentage. The trajectory forecast on the same
screen *does* give you a number, because that model validates.

If you ever see the decline model start issuing risks, it is because something
in the data actually predicts decline.

---

## Putting your own data in

Two ways, for two different jobs.

### A full cohort workbook

**Data → Upload a cohort workbook.** An Excel file with sheets for Baseline and
follow-up visits. Dive matches columns by name, so it tolerates most naming
conventions. Check the *Cleaning actions* tab afterwards.

To make real data the permanent source rather than a one-off upload:

1. Put the file in `data/real/` as `cohort_biomarkers.xlsx`
2. Edit `settings.yaml`: change `data_source: synthetic` to `data_source: real`

That is the only change. Filenames are configurable in `dive/config.py` under
`PathConfig` if yours differ.

### A single extra measurement

**Training and data → Add data.** A CSV, Excel or JSON file with a participant
ID column and whatever you measured. Dive maps the columns, accepts what is
plausible, and quarantines what is not — a grip strength of 250 kg is a unit
error, and it tells you rather than learning it.

---

## Adding a measurement Dive does not know

**Training and data → Health metrics → Register a new metric.**

Give it a name, a unit, and a plausible range. The range is the unit-error
detector: anything outside it gets quarantined on ingestion.

It becomes ingestable immediately. It does **not** go into any model — that is
the next step, deliberately.

---

## Making the model better

**Training and data → Metric value → Test metric value.**

This asks whether any available metric improves prediction on participants the
model has not seen. Most of the time the answer is no, and it says so.

It is strict on purpose. Try twenty metrics, keep the best, and you will find an
apparent improvement even when all twenty are noise. So it shuffles the outcome
and re-runs the *entire search* to see how large a gain that search finds when
nothing is real. On this data, a metric drawn from a random number generator
raised AUC by +0.098 and was correctly rejected.

**Retrain** fits a new model and promotes it only if it beats the current one on
held-out participants. **Autonomous** runs the whole cycle unattended.

---

## Imaging: calibrate before you trust it

**Training and data → Calibration.**

Register each scanner, record phantom QC scans. Dive flags drift and refuses to
pool measurements that are not comparable.

The number that matters is the **least significant change**. With typical pQCT
precision it is around 10 mg/cm³ — so a follow-up trabecular density 6 mg/cm³
lower has not changed, it is machine noise. Without this, a multi-site model
learns the scanner instead of the patient.

---

## Turning on the evidence advisor (optional)

The Recommendations screen has a curated evidence library that works offline.
The advisor adds participant-specific reasoning over retrieved literature, and
needs a key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or
export GEMINI_API_KEY=AIza...            # or
export OPENAI_API_KEY=sk-...
./start-dive-mac-linux.sh
```

By default the provider never sees your patients. It receives a literature
question — *"what does the evidence show about interventions for low grip
strength and frailty in older women?"* — and Dive matches the answer to the
participant locally. That query is identical for any two people who cross the
same thresholds.

Check this for yourself: **Privacy and security → What leaves**. Pick any
participant and see exactly what would be sent and what is withheld.

**This has not been tested against the live APIs.** The environment it was built
in blocks PubMed and had no keys. The citation-verification logic is tested
offline. Run it once with a key before relying on it.

---

## The owner console

This is the one screen only you get into. It needs a key, which you generate
once:

```bash
python -c "from dive.owner import OwnerGate; g=OwnerGate(); k=g.generate_key(); print('KEY :', k); print('HASH:', g.hash_key(k))"
```

Keep the **KEY** somewhere safe — it is shown once and Dive never stores it. Put
the **HASH** in your environment:

```bash
export DIVE_OWNER_KEY_HASH='scrypt$32768$...'
./start-dive-mac-linux.sh
```

Four tabs:

**What happened** — one chronological record across everything: data ingested,
outbound calls made or blocked, learning cycles, model promotions, source edits.
This is where *"why does it do that now?"* gets answered.

**What it learned** — every model version and its metrics, performance over
time, which metrics were tested and which earned a place, and every autonomous
cycle. A flat performance line is the expected result when new data carries no
new signal.

**Source** — read and edit the code, with a diff and an automatic backup.

**State** — audit chain integrity, egress mode, whether protected features are
allowed.

### Source editing is off by default, deliberately

Editing source through a web interface is remote code execution by design.
Anyone with your key can write arbitrary Python into the application. A path
allowlist narrows the blast radius; it does not remove it.

To turn it on, set `allow_source_editing: true` in `settings.yaml`. Even then:

- writes are confined to `dive/`, `app/`, `tests/`, `docs/` — never `.git`,
  `.venv`, `data/`, or any settings file holding secrets
- Python is parsed before it is written; a file that would not import is
  refused rather than saved and discovered at the next restart
- a timestamped backup is taken before every save, and any version restores
- every read, write and restore is audited with before-and-after hashes
- nothing is executed — changes take effect at the next restart, so a bad edit
  is a failed start rather than a live compromise

**If Dive is reachable from anything other than localhost or a network you
trust, leave this off** and edit the repository the normal way.

---

## If something goes wrong

**Port already in use** — something else is on 8501:
`./start-dive-mac-linux.sh` then `./.venv/bin/python -m streamlit run app/app.py --server.port 8502`

**Start completely fresh** — delete `.venv` and run `./start-dive-mac-linux.sh` again.

**Blank or stale screen** — refresh the browser; Streamlit caches aggressively.

**Something looks wrong in the numbers** — run `./start-dive-mac-linux.sh check`. If the 17
validation checks pass, the pipeline is behaving; if they fail, the output names
which check and why.

---

## Where to read more

| Document | Covers |
|---|---|
| `README.md` | What Dive is and how it is put together |
| `docs/OMICS.md` | All 137 metrics, and the measurement traps in each domain |
| `docs/SECURITY.md` | Privacy boundary, security controls, and what is not covered |
| `docs/MODELLING.md` | Why the models are built the way they are |
