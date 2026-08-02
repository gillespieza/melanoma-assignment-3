# The Melanoma Digital Twin Dashboard — Complete User Guide

**Who this is for:** anyone opening this dashboard for the first time, with no
assumed background. Every term is explained the first time it appears, and
there's a glossary at the end for quick lookup. Nothing in this guide assumes
you've read any other document in this repo.

---

## 1 · What is this thing, in one paragraph

This is a web page that helps a cancer clinician (an oncologist) decide which
drug to give a patient with **melanoma** (skin cancer) that has spread beyond
the skin. Instead of one method giving one answer, this tool runs **four
different, independent scientific methods** on the same patient and shows the
clinician where they agree and where they disagree. It never makes the
decision for the clinician — it presents ranked options with the evidence
behind each one, and the clinician picks and signs off on the plan. Everything
on screen is built
from **421 real, de-identified patient records** from a public cancer genomics
database (more on that below) — nothing is invented or simulated from
scratch.

Think of it like a second (and third, and fourth) opinion, generated
instantly from data, that a consultant can accept, adjust, or overrule.

---

## 2 · The two pages, and how you move between them

The whole tool is one webpage that switches between two "views." Look at the
top-right of the header — there are two tabs: **Cohort** and **Patient**.

- **Cohort** — the home page. A big searchable table of all 421 patients.
- **Patient** — one specific patient's full workup. You get here by clicking
  a row in the Cohort table (or one of the three "Featured Cases" cards).

The web address (URL) changes as you navigate — e.g.
`#/patient/TCGA-D3-A1Q1` — so you can bookmark or share a link straight to one
patient, and refreshing the page won't lose your place.

---

## 3 · The Cohort page, top to bottom

When the page loads, you'll see, in this order:

### 3.1 The headline strip
A white bar with a sentence explaining the tool, plus four number tiles:
**Patients** (421), **BRAF V600** (how many carry that specific mutation —
explained below), **Methods concordant** (how many patients had all methods
agree), **Methods split** (how many had methods disagree — these are the
interesting/hard cases worth a clinician's attention).

### 3.2 The methods strip
Five small tiles in a row, showing the five things this project does and how
they connect:

| Tile | Plain-English meaning |
|---|---|
| **ML predictor** | A statistics/machine-learning model that reads a tumour's gene activity and estimates the odds it'll respond to immunotherapy |
| **Validation** | Checking that ML predictor against real lab measurements it never saw, to make sure it isn't just guessing |
| **ODE digital twin** | A mathematical simulation of how *this specific patient's* tumour would shrink or grow under different drugs |
| **Resistance** | Rules-of-thumb flagging why a treatment might eventually stop working, and what to try next |
| **Integration** ("You are here") | Combining all four of the above into one final ranked recommendation |

These five map directly onto the five tabs you'll see later on a patient's
page (Q1, Q2, Q3, Q4, and the overall "Q5" recommendation).

### 3.3 Featured Cases
Three highlighted example patients, picked automatically because each one
cleanly demonstrates a classic decision-making scenario:
- **A — BRAF-mutant, PD-L1 low:** has a drug-targetable mutation, but the immune system looks "cold" (unlikely to respond to immunotherapy alone) → expect targeted therapy to win.
- **B — BRAF wild-type ("normal"), PD-L1 high:** no targetable mutation, but immune-"hot" biology → expect immunotherapy to win.
- **C — BRAF-mutant, PD-L1 high:** both options look viable → the genuinely hard sequencing decision.

These aren't fake demo patients — they're real people from the cohort,
re-selected every time the page loads based on whoever currently best fits
each pattern.

### 3.4 The cohort table
A sortable, filterable, searchable table of all 421 patients. Columns:

- **Patient** — the anonymised TCGA patient ID (e.g. `TCGA-D3-A1Q1`). This code is how the public cancer genomics project (TCGA, explained below) refers to this de-identified person.
- **BRAF** — whether they carry the BRAF V600 mutation ("V600") or not ("WT" = wild-type = normal/unmutated).
- **NRAS** — same idea, for a different gene.
- **PD-L1** — a lab measurement (band: Low/Intermediate/High) that roughly predicts how visible the tumour is to the immune system, plus its percentile rank against the rest of the cohort.
- **Response evidence** — a 0–100 score summarising how likely this patient is to respond to immunotherapy, combining whichever data sources are available.
- **Anti-PD-1 ↓** and **BRAFi ↓** — how much the tumour shrank in the computer simulation for each drug class (see §5.6). A dash (—) means the simulation had nothing usable to say for that patient (see the "uninformative twin" caveat in §8).
- **Q5 recommendation** — the tool's top-ranked option for this patient, with a confidence percentage.
- **Agreement** — whether the statistical method and the mechanistic simulation landed on the same answer (Concordant/Partial/Discordant/Single method).

Above the table are filter dropdowns (BRAF status, PD-L1 band, methods
agreement, and whether a real treatment record exists for that patient — see
§8), a search box, and a sort control. Click any row to open that patient.

### 3.5 The Q1 model-accuracy panel
At the very bottom of the Cohort page (only here — not on the patient page)
sits a technical panel showing how accurate the ML predictor actually is,
tested against real outcomes. This is explained in detail in §6.1.

---

## 4 · Opening a patient — what you'll see first

Click any patient row. You land on their page, which always starts with:

### 4.1 The Patient Passport
A card with this patient's real, recorded data:

- **Age**, **Sex**, **Stage** (how far the cancer has spread, using the standard I–IV oncology staging system — I is earliest, IV is most advanced/metastatic), **Follow-up** (how many months this patient was tracked for, and whether they died during that window or were still alive when tracking stopped — "censored").
- **PD-L1 (CD274)** and **PD-1 (PDCD1)** — two related but different genes. PD-1 is a "brake" protein on immune T-cells; PD-L1 is the matching "off switch" tumours often display to hide from those T-cells. Both are shown as a percentile against the rest of the cohort (e.g. "72nd pct" means higher than 72% of other patients).
- **TMB** (Tumour Mutational Burden) — how many mutations per megabase of DNA the tumour carries. Roughly speaking, more mutations = more chances the immune system can recognise the tumour as foreign, so high TMB often (not always) predicts better immunotherapy response.
- **MAPK driven** — whether this tumour's growth is being driven by an overactive MAPK signalling pathway (a chain of proteins inside the cell that tells it to divide) — BRAF and NRAS mutations both work by permanently switching this pathway on.
- **Treatment received** — what this patient was *actually* given, historically, according to TCGA's records (only available for about 47% of patients — TCGA simply didn't record it for the rest). Grouped by treatment type (e.g. "Targeted Molecular Therapy") with the specific drug underneath (e.g. "Vemurafenib"). Where the patient received a checkpoint inhibitor or BRAF/MEK inhibitor, a note explicitly says this is comparable to the two arms the digital twin simulates. Where they received chemotherapy or radiotherapy, a note explicitly says **neither is modelled by any method here** — melanoma chemotherapy today is largely palliative (symptom relief), not something with a proven survival benefit, so it's shown as historical context only, never as a comparator.
- Three coloured pills in the top-right: **BRAF** status, **NRAS** status, and **PD-L1 band** (High/Intermediate/Low).

If you've used the What-If explorer (next), this card gets an **amber
"Modified"** badge and every field it changed shows "· edited."

### 4.2 The What-If explorer
A row of controls: BRAF, NRAS, Stage, LDH, and a PD-L1 slider. Change any of
them and **every method on the page instantly recalculates** — this is the
tool's core "what if this patient's biology had been different?" feature. A
few important honesty notes baked into the design:

- **LDH is a field TCGA never recorded at all.** Lactate dehydrogenase is a
  blood enzyme; elevated LDH is a red flag for rapidly progressing disease
  and normally pushes clinicians toward faster-acting treatment. Since it's
  missing from every single patient in this dataset, the tool quietly uses
  "Stage IV" as a stand-in signal for that same urgency — until you supply an
  actual LDH value in the what-if bar, which then takes over.
- The moment you change BRAF, NRAS, or PD-L1, this patient's *own* simulated
  dose-response curve (§5.6) is no longer valid — the biology has changed. So
  the tool swaps in the closest matching real subgroup-average curve instead
  of pretending the original curve still applies, and says so on screen (a
  banner reading *"Modified profile — showing the matching subgroup average,
  not this patient's own simulation"*).
- A **"Restore real patient"** button appears once you've changed anything,
  to snap back to the true recorded values.

### 4.3 "Run Digital-Twin Simulation"
Before you've clicked this button, nothing else shows. Click it, and a short
animated sequence plays through the simulation steps, then the whole rest of
the page appears at once: the recommendation, the agreement badge, the ranked
options, and the tab bar. This isn't just a visual flourish — it genuinely
triggers the calculation described in §5.

---

## 5 · The recommendation area (after you run the simulation)

### 5.1 Integrated recommendation banner
A single sentence at the top: which treatment lane wins, the model's
confidence percentage, and the predicted median survival in months for that
choice. This is the tool's headline answer.

### 5.2 The Agreement Badge
This is arguably the most important card on the page. Two completely
independent methods each form their own opinion on whether immunotherapy will
work for this patient:

- **Statistical** — the data-driven read: either a real machine-learning
  prediction (if available) or, for these TCGA patients (where the ML model
  hasn't been run per-patient yet), a stand-in composite of the patient's
  measured PD-L1, PD-1, and TMB percentiles. The card always says which one
  it's using.
- **Mechanistic** — the digital-twin simulation's own read: how much tumour
  burden the anti-PD-1 simulation achieved for this specific patient.

The badge is colour-coded by how well these two agree:
- 🟢 **Concordant** — both methods land on the same side, strongly (≥70% agreement).
- 🔵 **Partial** — both point the same direction but with different strength.
- 🟠 **Discordant** — they land on *opposite* sides. This is flagged explicitly as worth a full multidisciplinary team (MDT) discussion, because the data and the biology are telling different stories.
- ⚪ **Single method only** — the twin simulation had nothing usable to say for this patient (see §8), so only the statistical side could be scored, and the recommendation is explicitly lower-confidence as a result.

### 5.3 Ranked Treatment Options
Three cards, one per treatment lane:

1. **Immunotherapy** — checkpoint blockade (real drugs: Nivolumab + Ipilimumab). Works by releasing the immune system's own "brakes" so T-cells can attack the tumour.
2. **Targeted therapy** — BRAF/MEK inhibitors (real drugs: Encorafenib + Binimetinib). Works by directly blocking the overactive growth signal in BRAF-mutant tumours — only usable if the patient actually carries that mutation.
3. **Combination / sequencing** — start with immunotherapy, hold targeted therapy in reserve for if/when the tumour progresses (or occasionally a genuine triplet regimen).

Each card shows: a tier badge (**Primary recommendation** / **Alternative** /
**Not recommended**), a confidence percentage, predicted **median overall
survival** in months, the **12-month tumour burden reduction** the twin
projects, a plain-English rationale, one line of supporting published trial
evidence (e.g. *"COLUMBUS: encorafenib+binimetinib median OS 33.6 mo"*), and
a caution (e.g. toxicity risk, or resistance timeline). A consultant can
click **"Select & confirm this plan"** on any card — this is a UI-only
selection for the demo, staging that choice for the sign-off box at the
bottom of the page.

### 5.4 The five method tabs
Below the ranked options sits a row of five tab buttons. Only one panel shows
at a time — click a tab to switch. **The page opens on the Q3 tab by
default**, so if you're looking for something else, you need to click over.

---

## 6 · The five tabs, in full detail

### 6.1 Q1 · ML predictor
**What Q1 is:** a machine-learning model (specifically, an ensemble — a
"committee vote" — of five different statistical models: logistic
regression, random forest, XGBoost, a support vector machine, and an elastic
net) trained to evaluate 12 multimodal features (immune signatures, macrophage
barrier metrics, driver mutation flags, and tumour mutational burden) from a patient profile
and output a single number: the probability this patient responds to checkpoint
immunotherapy.

**On a patient's page**, if the model has scored this specific patient,
you'll see a circular gauge with their **P(response)** percentage, bars
breaking that down by individual model (where available), and the input
feature breakdown (immune signatures, macrophage spatial metrics, driver mutation
status, and TMB — all measuring different dimensions of tumour immunogenicity and
microenvironmental barrier strength).

**Important honesty note:** many patients only carry a single combined score
rather than the five-model breakdown, because the underlying data file only
supplied one number per patient for TCGA. The tab says so explicitly rather
than inventing fake per-model numbers.

**The separate, cohort-level accuracy panel** (only on the Cohort landing
page, §3.5) is a completely different thing: it's not about any one patient,
it's proof that the Q1 models actually work, tested against 195 patients from
three real immunotherapy clinical trials (Liu 2019, Riaz 2017, Hugo 2016)
where the *true* outcome (did they actually respond, confirmed by a clinician)
is known. Headline: **AUC 0.593** overall (0.766 in the Riaz trial
specifically, 0.451 — essentially chance — in the small Hugo trial). AUC
("Area Under the [ROC] Curve") is a standard 0–1 accuracy score for a
yes/no prediction: 0.5 = no better than a coin flip, 1.0 = perfect. 0.593 is
a real but modest signal, and the panel says so plainly, including that the
model's raw probabilities are poorly calibrated (it pushes most patients
above 50% even though the true response rate is much lower) — so the
*ranking* the model produces is more trustworthy than any single number.

### 6.2 Q2 · Validation
**What Q2 is:** proof, from data the models never saw during training, that
the underlying science actually holds up — not a per-patient result, the
*same* panel appears no matter which patient you're viewing. Four pieces of
evidence:

1. **RPPA proteomic validation** — the digital twin predicts a protein
   called pERK (a direct readout of MAPK pathway activity) purely from its
   equations; that prediction is checked against real, physically-measured
   protein levels (from a lab technique called reverse-phase protein array,
   RPPA) in the same 310 patients. Result: a real but modest correlation
   (r = 0.175, p = 0.002 — statistically real, not by chance, but not a
   strong relationship on its own).
2. **Rank agreement** — both the model and the real protein measurements
   agree on the *ordering*: NRAS-mutant tumours have the most MAPK activity,
   then BRAF-mutant, then normal — a much more convincing check than the raw
   correlation number alone (p = 3.2×10⁻⁹, an extremely small/significant
   p-value).
3. **Efficient signal** — the twin's three simulation outputs alone match
   the accuracy of a machine-learning model trained on twelve raw clinical
   features, evidence the simulation is capturing real biology rather than
   just curve-fitting.
4. **Cell-line drug-response validation** — a completely separate model
   (built by a different team member on a different dataset — melanoma
   *cell lines* grown in a lab dish, not patients) trained to predict how
   much a drug shrinks each cell line, purely from its gene activity. Of five
   drugs tested, only one — Dabrafenib (a BRAF inhibitor) — produced a
   defensible result, and even that comes with an honesty caveat spelled out
   on the card: its best single test showed a correlation of 0.75, but when
   the model was refit five separate times on different random splits of the
   same data (a standard way to check if a result is a fluke), the average
   correlation dropped to just 0.28, ranging as low as -0.29. The card
   deliberately keeps that full picture visible rather than only quoting the
   best number.

### 6.3 Q3 · Digital twin
**What a "digital twin" is, literally:** a mathematical model — a system of
equations describing how tumour cells, immune cells, and drug concentration
change over time (an ODE, or "Ordinary Differential Equation," is just a
type of equation describing a rate of change) — solved separately, in
advance, for **each of the 421 patients individually**, using their own real
molecular profile as the starting conditions. It's called a "twin" because
it's meant to behave like a virtual copy of that one specific patient's
tumour.

**On this tab you'll see:**
- Four stat tiles: how much the simulated tumour shrank under each drug (as
  a percentage of its untreated size), and the "optimal dose" — the point on
  the dose scale where more drug stops helping.
- A line chart with two curves (BRAF inhibitor in blue, anti-PD-1 in teal),
  x-axis = drug dose from 0–100% of maximum, y-axis = simulated tumour size
  as a percentage of the untreated baseline (100% = no effect, lower = more
  shrinkage).
- If the patient is BRAF wild-type, you may see a flat or *rising*
  BRAF-inhibitor line — this is a real, well-documented phenomenon called the
  **RAF paradox**: giving a BRAF inhibitor to a tumour that doesn't actually
  have the BRAF mutation can *accelerate* growth through a quirk of the same
  signalling pathway, and the simulation reproduces this correctly.
- Below the tab, a **12-month Tumour Burden Forecast** chart and a **Kaplan–
  Meier survival curve**. Kaplan-Meier is the standard way oncology reports
  survival: the y-axis is "% of patients still alive," the x-axis is time —
  it always starts at 100% and steps downward. The *endpoints* of these
  curves are anchored to the real simulation output; the *shape* between
  points and the survival curves themselves are honestly labelled as
  projections (Weibull-fitted curves anchored to real survival medians from
  the cohort), not raw model output.

**The one big caveat on this whole tab:** for about 21% of the 421 patients
(93 of them), the simulation settles at a mathematically-zero tumour from the
very start — the model has nothing usable to say for that patient. This is
explicitly **not** the same as "the drug doesn't work" (that would be a real,
meaningful finding) — it means the simulation itself has no signal, so the
tool shows a clear "Twin below model resolution" notice instead of a fake 0%.

### 6.4 Q4 · Resistance
**What this tab is:** a set of clinical rules-of-thumb (not a trained
model) flagging why the recommended treatment might eventually stop working,
and what to try if/when it does. The tab is explicitly badged
**"Heuristic"** — meaning rule-based, lower confidence than Q1/Q3, and the
panel says exactly that in a permanent footer note.

- **Resistance risk** badge (Low/Moderate/High/Not assessable), based on how
  much tumour shrinkage the twin achieved.
- **Escape mechanisms flagged** — plain-English warnings like *"NRAS-driven
  MAPK reactivation"* (the pathway can find another way back on even after
  BRAF is blocked) or *"Immune-cold (low PD-L1)"* (checkpoint resistance is
  likely from the start).
- **Hold in reserve** — an ordered list of salvage options to consider if
  the primary plan fails, e.g. a different targeted-therapy combination, or
  retrying checkpoint blockade if TMB is very high.

### 6.5 Decision path
A left-to-right flowchart of the exact logic the tool followed for this one
patient: Stage → BRAF status → PD-L1 band → whether the methods agreed →
final recommendation. This exists so the reasoning is fully auditable — a
consultant (or you) can trace exactly why the tool landed where it did,
step by step, rather than trusting a black box.

### 6.6 Consultant sign-off
At the very bottom of the page: a box showing whichever option you selected
in §5.3, and a **"Sign & finalise plan"** button. This is the tool's way of
staying honest about its own role — *"The model advises; the consultant
decides and is accountable for the plan."* Nothing is auto-committed; a human
has to actively confirm it.

---

## 7 · What's real, what's projected, and what's a rule of thumb

The dashboard is built around being explicit about this distinction
everywhere, rather than presenting everything with equal-looking confidence.
Here's the full picture in one place:

| Category | Examples | What it means |
|---|---|---|
| **Real, measured data** | Age, sex, stage, TMB, PD-L1/PD-1 expression, survival months, mutation status, treatment history (where recorded) | Comes straight from the patient's real clinical/genomic record. Not modelled. |
| **Real model output** | The Q3 dose-response curves, the Q1 accuracy figures, the RPPA correlation, the cell-line drug-sensitivity result | A model was actually run and produced this number; it's not invented, but it is a model's estimate, not a ground truth |
| **Projection** | The shape of the 12-month forecast between two known endpoints; the survival curves | The *endpoints* are real; the *curve connecting them* follows a standard, labelled statistical shape, not a patient-specific simulation |
| **Heuristic (rule of thumb)** | Everything in the Q4 tab | Written by hand from clinical knowledge, not learned from data. Useful as a prompt for discussion, not as a scored prediction |

---

## 8 · Known limitations — read this before trusting a number blindly

- **~21% of patients have an "uninformative" digital twin** (§6.3) — the
  simulation has nothing to say, not "the drug doesn't work."
- **Q1 has not been run per-patient on the TCGA cohort** for most of its
  detail — the gauge and percentile are real, but the five-model breakdown
  and full 12 input features usually aren't available (they exist for a
  different, labelled set of clinical-trial patients used purely for
  accuracy testing).
- **Only ~47% of patients have a recorded treatment history** — TCGA simply
  didn't collect this for everyone, and it varies enormously by which
  hospital submitted the data (some hospitals recorded it for every patient,
  others for none). Use the "Treatment" filter on the Cohort table to jump
  straight to the patients who have it.
- **LDH and ECOG performance status were never recorded by TCGA at all** —
  the what-if explorer lets you supply a hypothetical LDH value, but every
  real patient in this cohort is missing it.
- **Chemotherapy and radiotherapy are shown for historical context only** —
  no method in this tool models either one. In modern melanoma care, both
  are largely used for symptom control (palliation) once checkpoint and
  targeted therapy have been tried, not because they extend survival — the
  literature doesn't support them as first-line, survival-directed options
  the way immunotherapy and targeted therapy are.
- **The cell-line drug validation (Q2) only has real coverage for one drug**
  (Dabrafenib) out of five tested, and even that result is unstable on
  resampling (§6.2) — treat it as suggestive, not proven.

---

## 9 · Glossary

**Clinical / biology terms**

- **Melanoma** — a cancer of melanocytes (pigment-producing skin cells); this project covers metastatic (spread beyond the skin) melanoma specifically.
- **TCGA / TCGA-SKCM** — The Cancer Genome Atlas, a large public US research programme that sequenced and catalogued the genomics of thousands of real, de-identified cancer patients. SKCM = Skin Cutaneous Melanoma, the specific dataset this whole cohort comes from.
- **BRAF / BRAF V600 mutation** — a gene that's part of the MAPK growth-signalling pathway. About 45% of melanomas carry a specific "V600" mutation that locks this gene permanently "on," driving uncontrolled growth — but also making it a precise drug target.
- **NRAS** — another gene in the same MAPK pathway; NRAS mutations also drive growth but aren't targetable by the same BRAF/MEK-inhibitor drugs.
- **Wild-type (WT)** — the normal, unmutated version of a gene.
- **MAPK pathway** — a chain of proteins inside a cell that relays a "grow and divide" signal; BRAF and NRAS mutations both jam this pathway in the "on" position.
- **PD-1 (gene: PDCD1)** — a protein on the surface of T-cells (immune attack cells) that acts as an "off switch," preventing them from attacking. Checkpoint immunotherapy drugs block this switch.
- **PD-L1 (gene: CD274)** — the matching partner protein, often displayed by tumour cells, that flips that T-cell "off switch." High PD-L1 usually predicts a better response to checkpoint immunotherapy.
- **TMB (Tumour Mutational Burden)** — the number of DNA mutations per megabase in a tumour. Higher TMB generally means more chances the immune system can recognise the tumour as abnormal.
- **Checkpoint immunotherapy / checkpoint inhibitor** — drugs (e.g. Nivolumab, Pembrolizumab, Ipilimumab) that block the PD-1/PD-L1 "off switch," letting T-cells attack the tumour.
- **Targeted therapy (BRAF/MEK inhibitor)** — drugs (e.g. Vemurafenib, Dabrafenib, Trametinib, Encorafenib, Binimetinib) that directly block the overactive MAPK signal in BRAF-mutant tumours specifically.
- **RAF paradox** — the counterintuitive effect where giving a BRAF inhibitor to a tumour *without* the BRAF mutation can actually speed up its growth, via a quirk of how the drug interacts with the pathway.
- **Chemotherapy (in this project: Dacarbazine, Temozolomide)** — older, non-targeted, non-immune drugs that kill rapidly dividing cells generally. In modern melanoma care, mostly used for symptom relief once other options are exhausted.
- **Stage (I–IV)** — the standard oncology system for how far a cancer has spread; IV is the most advanced (metastatic).
- **LDH (Lactate Dehydrogenase)** — a blood enzyme; elevated levels are a marker of rapidly progressing disease, used by clinicians to judge urgency.
- **ECOG performance status** — a standard 0–5 scale of how well a patient can function day-to-day; used alongside stage to judge how aggressive treatment can safely be.
- **Overall survival (OS) / median OS** — how long patients lived, measured from a fixed starting point; the *median* is the point where half the group has died and half hasn't — the standard way trials report survival.
- **Kaplan-Meier curve** — the standard statistical way to chart survival over time, correctly handling patients who are still alive when the study ends ("censored").
- **RPPA (Reverse-Phase Protein Array)** — a lab technique that measures the actual amount of specific proteins present in a tumour sample — used here as ground truth to check whether the model's predictions match physical reality.
- **pERK** — the "activated" (phosphorylated) form of a protein called ERK, a direct downstream readout of how active the MAPK pathway currently is.
- **RECIST response** — the standard clinical criteria clinicians use to classify whether a tumour shrank, grew, or stayed the same on a scan, used to label real clinical-trial patients as "responders" or "non-responders."

**Statistics / machine-learning terms**

- **AUC (in Q1's accuracy panel) = Area Under the ROC Curve** — a 0–1 score for how well a model separates two groups (here: responders vs non-responders). 0.5 = coin flip, 1.0 = perfect. *Careful: Q2's cell-line data also uses the word "AUC" to mean something completely different — see below.*
- **AUC (in the cell-line/Q2 context) = Area Under the [drug-response] Curve** — a lab measure of how much a drug killed a cell line, ranging roughly 0 (all cells died) to 1 (no effect). Same three letters, unrelated meaning — don't confuse the two.
- **ROC curve** — a chart plotting a model's true-positive rate against its false-positive rate across every possible decision threshold; the area underneath it is the AUC.
- **Sensitivity / Specificity** — sensitivity = what fraction of true responders the model correctly caught; specificity = what fraction of true non-responders it correctly ruled out.
- **PPV / NPV** — Positive/Negative Predictive Value: given a positive (or negative) prediction, how often is it actually correct.
- **Threshold (e.g. "tuned" vs "0.50")** — the cutoff probability above which a prediction counts as "positive." 0.50 is the naive default; a "tuned" threshold is chosen to work better for this specific population's true response rate.
- **Percentile** — where a value ranks compared to everyone else, 0–100. "72nd percentile" means higher than 72% of the cohort.
- **Correlation (r)** — a -1 to 1 number measuring how strongly two things move together. 0 = no relationship, 1 = perfect positive relationship, -1 = perfect inverse relationship.
- **p-value** — the probability that a result this strong could have happened purely by chance if there were actually no real effect. Smaller = more confidently a real effect (the usual cutoff is p < 0.05).
- **n** — the sample size (how many data points a statistic was calculated from). A correlation from n=5 is far less trustworthy than the same correlation from n=100, even if the number looks identical.
- **LASSO regression** — a statistical technique for building a prediction model when you have far more candidate input variables (e.g. ~19,000 genes) than data points (e.g. 37 cell lines) — it automatically ignores most inputs and keeps only the handful that matter most.
- **Z-score** — how many standard deviations above or below the group average a value sits. 0 = exactly average, positive = above average, negative = below average.
- **ODE (Ordinary Differential Equation)** — a mathematical equation describing how a quantity changes over time; here, used to simulate tumour cells, immune cells, and drug levels interacting continuously rather than in discrete steps.
- **Digital twin** — a simulation built to mimic one specific real-world entity (here: one patient's tumour) closely enough to be used for "what if" experiments on the computer instead of the real thing.

---

## 10 · Worked example — following one patient end to end

To make all of the above concrete, here's a plain-language walkthrough of a
real case discovered during this project (patient `TCGA-D3-A2JE`):

1. **Passport:** 75-year-old woman, Stage IIIC, BRAF V600E mutant, PD-L1 in
   the 2nd percentile (about as immune-cold as a tumour gets in this cohort).
2. **Digital twin (Q3):** the BRAF-inhibitor simulation clears the tumour
   almost completely — burden drops to 0.1% of baseline by the time the dose
   reaches about a third of maximum. The anti-PD-1 simulation shows a
   completely flat line — 0% response, consistent with how immune-cold the
   tumour is.
3. **Agreement badge:** because this patient is BRAF-mutant with a clear
   biological reason to expect targeted therapy to work and immunotherapy to
   fail, the statistical and mechanistic reads agree — this lands as a
   concordant case.
4. **Ranked options:** targeted therapy comes out as the clear primary
   recommendation, with immunotherapy correctly shown as unfavourable given
   the biology.
5. **The lesson:** this is *not* an example of "nothing works" — it's one of
   the cleanest, most confident calls the whole cohort produces. It's a good
   reminder that "immune-cold" doesn't mean "untreatable" — it means "don't
   reach for immunotherapy first," which is exactly the kind of call this
   tool exists to make quickly and defensibly.

---

## 11 · Who is "Dr. Aoife Gallagher"?

A fictional consultant. The name and initials shown in the top-right corner
of every page are not a real, identifiable person, and not tied to any
specific patient in the data — they exist purely so the sign-off flow (§6.6)
reads naturally in a demo.

---

*This guide describes the dashboard as of the `Dashboard_final` branch. If a
feature described here doesn't match what's on screen, the dashboard has
likely moved on since this was written — check `docs/07_SESSION_LOG.md` for
the most recent build notes.*
