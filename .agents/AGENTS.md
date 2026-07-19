# Project Style & Visualization Guidelines

Whenever generating plots, figures, web artifacts, or presentation reports for this project, always adhere to the project-wide visual style and Google Material Design color palettes defined below.

---

## 1. Color Palettes (Google Material Design System)

### Study Cohorts
- **Liu 2019**: `#1976D2` (Material Blue 700)
- **Hugo 2016**: `#E65100` (Material Deep Orange 900)
- **Riaz 2017**: `#673AB7` (Material Deep Purple 500)
- **TCGA-SKCM**: `#37474F` (Material Blue Grey 800)
- **Pooled Trials**: `#00796B` (Material Teal 700 Benchmark)

### Clinical Response & Phenotypes
- **Responder (CR / PR)**: `#2E7D32` (Material Green 800)
- **Non-Responder (PD)**: `#C62828` (Material Red 800)
- **Stable Disease (SD)**: `#F57C00` (Material Orange 700)

### Driver Mutation Subtypes
- **BRAF**: `#D32F2F` (Material Red 700)
- **NRAS**: `#0288D1` (Material Light Blue 700)
- **NF1**: `#388E3C` (Material Green 700)
- **Triple-WT**: `#7B1FA2` (Material Purple 700)

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
