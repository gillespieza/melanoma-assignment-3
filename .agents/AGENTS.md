# Project Style & Visualization Guidelines

Whenever generating plots, figures, web artifacts, or presentation reports for this project, always adhere to the project-wide visual style and Okabe-Ito (Cell / Nature / Science Gold Standard) color palettes defined below.

---

## 1. Color Palettes (Okabe-Ito Scientific Gold Standard)

### Study Cohorts
- **Liu 2019**: `#0072B2` (Okabe-Ito Blue)
- **Hugo 2016**: `#E69F00` (Okabe-Ito Orange)
- **Riaz 2017**: `#CC79A7` (Okabe-Ito Reddish Purple)
- **TCGA-SKCM**: `#37474F` (Dark Slate Charcoal Reference)
- **Pooled Trials**: `#009E73` (Okabe-Ito Bluish Green Benchmark)

### Clinical Response & Phenotypes
- **Responder (CR / PR)**: `#009E73` (Okabe-Ito Bluish Green)
- **Non-Responder (PD)**: `#D55E00` (Okabe-Ito Vermillion Red)
- **Stable Disease (SD)**: `#F0E442` (Okabe-Ito Yellow)

### Driver Mutation Subtypes
- **BRAF**: `#D55E00` (Okabe-Ito Vermillion)
- **NRAS**: `#56B4E9` (Okabe-Ito Sky Blue)
- **NF1**: `#009E73` (Okabe-Ito Bluish Green)
- **Triple-WT**: `#CC79A7` (Okabe-Ito Reddish Purple)

---

## 2. Matplotlib & Seaborn Defaults

- **Resolution**: Default to `dpi=300` for all saved PNG figures (presentation and publication quality).
- **Aspect Ratio**: Prefer 16:9 widescreen proportions (`figsize=(11, 6)` or `figsize=(12, 7)`) when figures are intended for slides.
- **Typography**: Clean `sans-serif` font with bold axis labels and clear title hierarchy.
- **Labels & Annotations**: On presentation slides, prefer direct data labels on bars/markers over forcing audience grid tracing.
- **Python Imports**: Import central color palettes from `src.styles` (`from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style`).

---

## 3. Language & Spelling Conventions

- **Spelling**: Always use **British English** spelling for all generated text, markdown reports, code docstrings, plot titles, labels, and presentation slide content (e.g., *colour*, *visualisation*, *characterisation*, *tumour*, *analyse*, *modelling*, *centre*).
