# Melanoma First-Line Treatment Triage — Clinician Guide

**Purpose.** A structured, research-grade decision aid for selecting first-line systemic
therapy in advanced cutaneous melanoma when multiple options are on the table
(checkpoint immunotherapy vs. BRAF/MEK-targeted therapy vs. chemotherapy). It combines two
independent tumour-biology readouts with mutation status.

> **Scope & disclaimer.** Decision *support*, not a validated clinical device or a
> substitute for guideline-based care. It does not replace multidisciplinary judgement,
> and it does not incorporate LDH, disease stage, brain-metastasis status, performance
> status, prior therapy, or patient preference — all of which can override the suggestion
> below. Use alongside current guidelines.

---

## Step 1 — Gather three inputs

| Input | How it's obtained | What it tells you |
|-------|-------------------|-------------------|
| **BRAF-V600 status** | Tumour NGS / mutation panel | Whether targeted therapy is an *option* at all |
| **Immune infiltration ("hot vs cold")** | Cytotoxic-T transcriptional score (CD8A / PRF1 / GZMA; or the full immune-response classifier) | Likelihood of checkpoint-immunotherapy benefit |
| **Disease pace / burden** | Modelled tumour burden; clinically: LDH, volume, symptoms, rate of progression | Whether rapid control is needed up front |

---

## Step 2 — Route on the 2 × 2 grid

```
                         BRAF-V600 MUTANT              BRAF WILD-TYPE (incl. NRAS)
                  ┌──────────────────────────────┬──────────────────────────────┐
   IMMUNE-HOT     │  IMMUNOTHERAPY FIRST          │  IMMUNOTHERAPY               │
   (inflamed)     │  (if high burden/symptomatic: │  (no targeted option;        │
                  │   targeted → immunotherapy)   │   checkpoint blockade)       │
                  ├──────────────────────────────┼──────────────────────────────┤
   IMMUNE-COLD    │  TARGETED THERAPY FIRST       │  TRIAL / CHEMOTHERAPY        │
   (excluded)     │  (BRAF/MEK inhibitor)         │  (hardest group)             │
                  └──────────────────────────────┴──────────────────────────────┘
```

---

## Step 3 — What each route means in practice

**Immunotherapy first** — anti-PD-1 ± anti-CTLA-4 (e.g. nivolumab ± ipilimumab, or
pembrolizumab). Rationale: highest chance of a *durable* response in immune-hot tumours;
in BRAF-mutant patients, targeted therapy is kept in reserve for later progression.

**Targeted → immunotherapy** — for BRAF-V600, immune-hot but high-burden/symptomatic
disease: a short course of BRAF+MEK inhibition (e.g. dabrafenib + trametinib) for rapid
cytoreduction, then transition to immunotherapy for durability.

**Targeted therapy first** — BRAF+MEK inhibitor (e.g. dabrafenib + trametinib, or
encorafenib + binimetinib) for BRAF-V600 tumours that are immunologically cold and less
likely to respond to checkpoint blockade. Reliable, rapid responses; typically less
durable, so plan the next line.

**Trial / chemotherapy** — BRAF-wild-type *and* immune-cold: the group with the fewest
good options. Prioritise **clinical trial** enrolment; consider a MEK inhibitor in
NRAS-mutant disease; chemotherapy (dacarbazine / temozolomide) is a palliative fallback.

---

## Step 4 — Clinical overrides (these beat the grid)

- **Symptomatic brain metastases / rapidly deteriorating** → prioritise the fastest
  reliable response (targeted therapy if BRAF-V600), regardless of immune status.
- **Contraindication to immunotherapy** (e.g. active severe autoimmune disease, organ
  transplant) → shift away from checkpoint blockade even if immune-hot.
- **BRAF-V600 + very high LDH / bulky disease** → favour the targeted-first or
  combination route for up-front control.
- **Patient preference / comorbidity / access** → always in scope.

---

## How the routes performed as prognostic strata (TCGA-SKCM, n = 421)

Median overall survival by assigned route — a check that the strata track real biology
(not a treatment-effect claim; see Q5 answer):

| Route | n | Median OS |
|-------|--:|----------:|
| Immunotherapy first | 198 | 117 mo |
| Targeted → immunotherapy | 13 | 111 mo |
| Targeted therapy first | 94 | 72 mo |
| Trial / chemotherapy | 116 | 50 mo |

Log-rank across groups: p = 2.7 × 10⁻⁵. The immune-hot routes carry the best prognosis and
the wild-type/immune-cold group the worst — the expected ordering.

---

*Generated from `scripts/triage.py`; see `Q5_answer.md` for methodology and limitations.*
