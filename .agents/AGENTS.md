# Project Style & Visualization Guidelines

Whenever generating plots, figures, web artifacts, or presentation reports for this project, always adhere to the project-wide visual style and color palettes defined below.

---

## 1. Color Palettes

### Study Cohorts
- **Liu 2019**: `#2B5C8F` (Slate Blue)
- **Hugo 2016**: `#D95F02` (Rust Orange)
- **Riaz 2017**: `#7570B3` (Purple)
- **TCGA-SKCM**: `#333333` (Charcoal)
- **Pooled Trials**: `#1B9E77` (Teal Benchmark)

### Clinical Response & Phenotypes
- **Responder (CR / PR)**: `#2E7D32` (Forest Green)
- **Non-Responder (PD)**: `#C62828` (Crimson Red)
- **Stable Disease (SD)**: `#F57C00` (Amber)

### Driver Mutation Subtypes
- **BRAF**: `#E41A1C` (Red)
- **NRAS**: `#377EB8` (Blue)
- **NF1**: `#4DAF4A` (Green)
- **Triple-WT**: `#984EA3` (Purple)

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

