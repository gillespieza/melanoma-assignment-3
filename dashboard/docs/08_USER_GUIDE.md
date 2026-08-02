# The Melanoma Digital Twin Dashboard – Complete User Guide

**Who this is for:** anyone opening this dashboard for the first time, with no
assumed background. Every term is explained the first time it appears, and
there's a glossary at the end for quick lookup. Nothing in this guide assumes
you've read any other document in this repo.

---

## 1 · What is this thing, in one paragraph

This is a web page that helps a cancer clinician (an oncologist) decide which
drug to give a patient with **melanoma** (skin cancer) that has spread beyond
the skin. Instead of relying on a single biomarker or model, this tool runs **a
multi-method scientific framework** (Q1–Q5) on the same patient: combining
**machine-learning immune predictors (Q1)**, **cell-line viability translational models (Q2)**,
**ODE mechanistic digital twins (Q3)**, **DepMap/LINCS CRISPR target discovery (Q4)**, and
a **Two-Stage GMM Patient Stratification Engine (Q5)**. It computes a continuous **Treatability Index (0–100)**,
assigns every patient to one of four biologically distinct phenotypes, and presents ranked options
with full evidence transparency. Everything on screen is built from **421 real, de-identified patient records**
from TCGA-SKCM – nothing is synthetic or invented from scratch.

Think of it like a second (and third, and fourth) opinion, generated
instantly from data, that a consultant can accept, adjust, or overrule.

---

## 2 · The two pages, and how you move between them

The whole tool is one webpage that switches between two "views." Look at the
top-right of the header – there are two tabs: **Cohort** and **Patient**.

- **Cohort** – the home page. A big searchable table of all 421 patients, paired with archetype cards and phenotype breakdown.
- **Patient** – one specific patient's full workup. You get here by clicking
  a row in the Cohort table (or one of the four "Subgroup Stratification Archetypes" cards).

The web address (URL) changes as you navigate – e.g.
`#/patient/TCGA-D3-A1Q1` – so you can bookmark or share a link straight to one
patient, and refreshing the page won't lose your place.

---

## 3 · The Cohort page, top to bottom

When the page loads, you'll see, in this order:

### 3.1 The headline strip
A white bar explaining the tool, plus four key metric tiles:
- **Patients** (421 total TCGA-SKCM digital twins)
- **Q5-scored** (421 with full Two-Stage GMM phenotyping & Treatability Index)
- **High confidence** (129 patients assigned High recommendation confidence)
- **Methods split** (how many had statistical vs. mechanistic methods disagree – these are the discordant cases requiring MDT review)

### 3.2 The Q5 phenotype summary strip
A horizontal distribution bar showing the benchmark proportion of the four Q5 phenotypes across the cohort:
- **Immune Hot** (36.6% – Inflamed, high TIS & CYT)
- **Immune Cold** (6.4% – T-cell desert, depleted infiltration)
- **M2-High** (48.8% – Stromal exclusion by M2 macrophages)
- **Mutant-Driven** (8.2% – 100% NF1 loss-of-function & high TMB)

### 3.3 The methods strip
Tiles showing the five core project questions and how they connect into the master Q5 synthesis:

| Question | Method & Clinical Role |
|---|---|
| **Q1 ML predictor** | Machine-learning ensemble (LR, RF, XGB, SVM, ENet) predicting P(response) to checkpoint blockade |
| **Q2 Validation** | Translational cell-line drug sensitivity models (GDSC2) & proteomic RPPA pERK validation |
| **Q3 Digital twin** | Mechanistic Ordinary Differential Equation (ODE) simulating patient-specific dose-response dynamics |
| **Q4 Target discovery** | DepMap CRISPR essentiality & LINCS L1000 perturbagen recommendations for resistance escape |
| **Q5 Stratification ("You are here")** | Master integration engine: Two-Stage GMM phenotyping, Treatability Index, and ranked therapy arms |

### 3.4 Subgroup Stratification Archetypes (Featured Cases)
Four highlighted example cases, representing the four Q5 biological phenotypes:
- **Immune Hot:** Inflamed microenvironment – primary candidate for **Immunotherapy** (Arm A).
- **Immune Cold:** T-cell desert – requires dual M2-depleting agent + checkpoint **Combination rescue arm** (Arm C).
- **M2-High:** Stromal exclusion by M2 macrophages – **Targeted therapy** (Arm B) or stromal remodelling.
- **Mutant-Driven:** 100% NF1 loss-of-function, RAS hyperactivation, high TMB – **ICI + MEK adjunct**.

### 3.5 The cohort table
A sortable, filterable, searchable table of all 421 patients. Columns:

- **Patient** – anonymised TCGA patient ID (e.g. `TCGA-D3-A1Q1`).
- **Q5 Phenotype** – colour-coded dot and short label (Immune Hot, Immune Cold, M2-High, Mutant-Driven).
- **Phenotype confidence** – Q5 GMM phenotype assignment confidence band (**High**, **Moderate**, **Low**). Reflects how clearly the patient sits inside their assigned cluster based on the GMM posterior probability across all four subgroups. High means the model is unambiguous; Low means the patient falls near a cluster boundary. This is the more principled confidence metric — it comes directly from the statistical model.
- **TI (Treatability Index)** – 0–100 quantile-scaled composite score quantifying overall therapeutic tractability by combining antigen presentation capacity, IFN-γ signaling, and microenvironmental barriers (macrophage STV score, stromal CAF exclusion). Hovering over the **TI** header in the patient table reveals an interactive popover hint explaining this metric.
- **ICI tumour ↓** and **BRAFi ↓** – projected percentage tumour burden reduction under each treatment arm, as simulated by the Q3 ODE digital-twin model. ICI refers to Anti-PD-1 checkpoint immunotherapy; BRAFi refers to BRAF inhibitor therapy (e.g. Dabrafenib + Trametinib). A dash (–) indicates the simulation was uninformative for that patient. Hovering over either column header reveals a popover with this explanation.
- **Q5 recommendation** – top-ranked treatment arm (Arm A: Immunotherapy, Arm B: Targeted, Arm C: Combination).
- **Rec. confidence** – 0–100 heuristic score from the integration engine reflecting how strongly the recommended arm scores over the alternatives for this patient. Inputs include PD-L1 percentile, IFN-γ signature, BRAF status, LDH, ECOG performance status, and Q3 ODE tumour reduction. This is a relative ranking signal, not a probability of clinical response — see **Phenotype confidence** for the statistically grounded certainty estimate.
- **Agreement** – concordance status between statistical and mechanistic models (Concordant, Partial, Discordant, Single method).

Above the table are controls: Search patient ID, Phenotype filter dropdown, Agreement filter, Treatment history filter, and Sort options (including Treatability Index).

### 3.6 The Q1 model-accuracy panel
Located at the bottom of the Cohort page, detailing Q1 ML performance across 195 trial patients (Liu, Riaz, Hugo) with ROC-AUC breakdown (overall AUC 0.5935, Riaz AUC 0.7659).

---

## 4 · Opening a patient – what you'll see first

Click any patient row to open their workbench:

### 4.1 The Patient Passport
Displays demographic, clinical, and Q5 stratification metadata:
- **Age**, **Sex**, **Stage** (I–IV), **Follow-up** months & vital status.
- **Molecular profile:** PD-L1 (CD274) %, PD-1 (PDCD1) %, TMB (mut/Mb), MAPK driven status, BRAF/NRAS mutation status.
- **Treatment received:** historical TCGA treatment records (where available).
- **Header Pills:** Q5 Phenotype (with Okabe-Ito color dot), Confidence Band (High/Moderate/Low), and Treatability Index (`TI 72/100`).

### 4.2 The What-If explorer
Allows clinicians to adjust BRAF, NRAS, Stage, LDH, and PD-L1 values to simulate hypothetical patient variants in real-time.

### 4.3 "Run Digital-Twin Simulation"
Triggers the multi-method synthesis, displaying the integrated recommendation, agreement badge, ranked options, and method detail tabs.

---

## 5 · The recommendation area

### 5.1 Integrated recommendation banner
Headline summary showing the primary recommended therapy arm, model confidence percentage, and predicted median overall survival (OS).

### 5.2 The Agreement Badge
Evaluates concordance between statistical evidence (Q1/Biomarkers) and mechanistic ODE simulations (Q3):
- 🟢 **Concordant** – both methods agree (≥70% concordance).
- 🔵 **Partial** – same direction, differing magnitude.
- 🟠 **Discordant** – methods split; flags case for multidisciplinary team (MDT) review.
- ⚪ **Single method only** – ODE twin settled at uninformative baseline.

### 5.3 Ranked Treatment Options
Presents three structured therapy options:
1. **Arm A: Immunotherapy** (Anti-PD-1 monotherapy or dual CTLA-4/PD-1 blockade).
2. **Arm B: Targeted Therapy** (BRAF/MEK inhibitors, e.g. Dabrafenib + Trametinib).
3. **Arm C: Combination / Reversal** (Immuno-rescue or sequential targeted + checkpoint).

Each card displays tier badge (**Primary**, **Alternative**, **Not recommended**), confidence %, median OS (months), 12-month burden reduction %, clinical rationale, trial evidence, and Q4 nominated targets/cautions.

---

## 6 · Method Detail Tabs

Click between six method tabs to inspect individual method outputs:

### 6.1 Q5 · Stratification (Primary View)
- **Assigned Phenotype:** assigned cluster label, short label, and cluster ID.
- **Treatability Index:** 0–100 quantile-scaled composite score.
- **GMM Cluster Membership Probabilities:** soft-weighted bar chart across all 4 subgroups (Immune Hot, Immune Cold, M2-High, Mutant-Driven).
- **Primary Treatment Arm & Q4 Target:** recommended regimen and Q4 nominated salvage target.
- **Phase 4 ODE Dynamic Trajectories:** endpoint residual tumour burden across therapy arms for this phenotype.

### 6.2 Q1 · ML predictor
P(response) probability gauge, 5-model ensemble breakdown (LR, RF, XGB, SVM, ENet), and 11 multimodal feature values (immune signatures, macrophage barrier & driver mutations).

### 6.3 Q2 · Validation
Proteomic RPPA pERK validation (r = 0.175, p = 0.002), rank order significance (p = 3.2×10⁻⁹), and GDSC2 cell-line viability calibration.

### 6.4 Q3 · Digital twin
10-step ODE dose-response sweep chart (BRAFi vs. Anti-PD-1), RAF paradox visualization, 12-month tumour burden forecast, and Kaplan-Meier survival curves.

### 6.5 Q4 · Resistance
Rule-based resistance risk assessment, escape mechanisms, and ordered reserve/salvage targets.

### 6.6 Q5 Enhanced ML Predictor
- **Assigned Subgroup Model:** phenotype-specific Random Forest model trained within cluster boundaries.
- **33-Feature Enriched Panel:** incorporates Macrophage STV Score, CAF Exclusion, M1/M2 Ratio, Antigen Presentation (APM), `NF1` loss-of-function, `BRAF/NRAS` status, and TMB.
- **Head-to-Head Comparison:** side-by-side matrix comparing the global 6-feature Q1 model against the Q5 33-feature subgroup model.
- **Feature Importance Profile:** top predictive feature weights for the patient's assigned subgroup model.
- **Empirical Evaluation Matrix:** cross-validation metrics (`ROC-AUC`, `PR-AUC`, `Precision`, `Recall`, `F1`, `Accuracy`, `Brier Score`) across all 4 phenotypes.

### 6.7 Decision path
Flowchart tracing patient decision tree logic: Stage/LDH → BRAF/NRAS → Q5 Phenotype → Methods Agreement → Recommendation.

---

## 7 · Glossary

- **Q5 Two-Stage GMM Phenotyping** – Unsupervised machine-learning stratification combining 3-component Gaussian Mixture Model (GMM) continuous immune clustering with Stage 2 deterministic genomic driver classification (`NF1` loss-of-function).
- **Treatability Index (TI)** – Quantile-scaled 0–100 score quantifying overall therapeutic tractability by combining antigen presentation, IFN-γ signaling, effector cell infiltration, and microenvironmental barriers.
- **Phenotype confidence** – GMM posterior-probability-derived band (High/Moderate/Low) reflecting how unambiguously the Q5 model assigns a patient to their phenotype cluster. Distinct from the arm confidence % shown next to the therapy recommendation, which is a heuristic scoring signal.
- **Immune Hot** – Inflamed phenotype with high TIS and CYT signatures; primary candidate for immunotherapy.
- **Immune Cold** – T-cell desert phenotype with low immune infiltration; candidate for Arm C combination rescue.
- **M2-High** – Immunosuppressive phenotype dominated by M2 macrophages and stromal exclusion; candidate for targeted therapy or M2-depleting agents.
- **Mutant-Driven** – Phenotype characterized by 100% NF1 loss-of-function, RAS hyperactivation, and high TMB; candidate for ICI + MEK adjunct.
- **Q4 Nominated Target** – Reserve target identified via DepMap CRISPR essentiality and LINCS L1000 perturbagen screens to overcome treatment resistance.
- **Okabe-Ito Palette** – Scientific colorblind-safe color standard used throughout all project visualizations and dashboard UI elements.

---

*This guide reflects the OncoTwin dashboard following complete Q5 integration.*
