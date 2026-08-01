# Project Style & Visualization Guidelines

Whenever generating plots, figures, web artifacts, or presentation reports for this project, always adhere to the project-wide visual style and Okabe-Ito (Cell / Nature / Science Gold Standard) color palettes defined below.

## 0. Project Architecture Map (READ FIRST)

- **Before exploring the codebase**, always read [`PROJECT_MAP.md`](../PROJECT_MAP.md) in the repository root. It contains the complete subproject inventory, phase-to-file mappings, module dependency graph, data flow paths, and known technical debt. This eliminates redundant file-discovery across conversations.
- **After completing any structural change** (new scripts, moved files, new modules, refactored phases, resolved tech debt), update `PROJECT_MAP.md` to reflect the current architecture before closing the task.

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

### Biological Phenotype Subtypes (`PHENOTYPE_PALETTE`)
- **Immune Hot**: `#D55E00` (Crimson / Vermillion Red)
- **Immune Cold**: `#0072B2` (Okabe-Ito Blue)
- **Immunosuppressive M2-High**: `#CC79A7` (Okabe-Ito Reddish Purple)
- **Mutant-Driven**: `#F0E442` (Okabe-Ito Yellow)

---

## 2. Matplotlib & Seaborn Defaults

- **Resolution**: Default to `dpi=300` for all saved PNG figures (presentation and publication quality).
- **Aspect Ratio**: Prefer 16:9 widescreen proportions (`figsize=(11, 6)` or `figsize=(12, 7)`) when figures are intended for slides.
- **Typography**: Clean `sans-serif` font with bold axis labels and clear title hierarchy.
- **Labels & Annotations**: On presentation slides, prefer direct data labels on bars/markers over forcing audience grid tracing.
- **Gridlines**: Set gridlines to thin, light gray (`grid.color: '#E5E7EB'`, `grid.linewidth: 0.5`, `grid.alpha: 0.6`) in `set_presentation_style()` so gridlines never distract from data visualisations.
- **Image Preference over Tables**: For presentation and reporting purposes, always prefer generating and embedding high-resolution (300 DPI) visual plots and figures over static tabular data.
- **Python Imports**: Import central color palettes from `src.styles` (`from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style`).

---

## 3. Language & Formatting Conventions

- **Spelling**: Always use **British English** spelling for all human-readable output text (e.g., *colour*, *visualisation*, *characterisation*, *tumour*, *analyse*, *modelling*, *centre*). The scope of this rule is:
  - ✅ **In scope** — apply British spelling to: markdown report body text and headings; code **comments** (`# ...`); **docstrings** (`"""..."""`); **plot title and axis label strings** (e.g. `set_title(...)`, `set_xlabel(...)`, `ax.text(...)`); **print/log output strings** and `f.write(...)` report content.
  - ❌ **Out of scope** — do **not** apply British spelling to: Python **function argument names or kwarg values** (e.g. `ha="center"`, `va="center"`, `loc="lower center"`, `color=...`, `edgecolor=...` — these are library API tokens and must match exactly what the library expects); **YAML key strings** in frontmatter or cssclass lists (e.g. `table-center` is the Obsidian CSS class name and must not be changed); **Python dictionary key strings** used as internal code identifiers (e.g. `{"color": "#aabbcc"}`); **import lines and module/class names** (e.g. `from matplotlib.colors import ...`); **variable names, loop variables, and function parameter names** (e.g. `for color in palette`, `return color`).
- **Gene Names**: Always wrap all gene names and gene symbols in backticks (e.g., `CD274`, `PDCD1`, `BRAF`, `BRAF V600`, `BRAF V600E`, `NRAS`, `NF1`, `B2M`, `TAP1`, `JAK1`, `STAT1`) across all generated markdown files, documentation, reports, and text artifacts.
- **Horizontal Rules**: Never use horizontal rules (`---` or `***`) in markdown files or responses unless explicitly instructed to do so (the project's custom CSS automatically renders horizontal rules beneath `<h1>` and `<h2>` headings).


---

## 4. Code Smells & Refactoring Guidelines

When fixing code smells or refactoring code in this repository, follow these guidelines:

1.  **Unused Imports & Variables**: Remove all unused imports and variables. Ensure your imports are organized according to standard PEP 8 conventions.
2.  **Long Functions**: Break down functions longer than 30 lines into smaller, well-named helper functions that do one thing well (Single Responsibility Principle).
3.  **Complex Conditionals**: Extract complex nested `if/else` logic into well-named boolean variables or separate evaluation functions to improve readability.
4.  **Magic Numbers/Strings**: Replace hard-coded values and magic numbers with descriptive, uppercase module-level constants (e.g., `MAX_ITERATIONS = 100`).
5.  **Type Hinting**: Add Python type hints (`->`, `:`, `List`, `Dict`, etc.) to all function signatures to make inputs and outputs explicit.
6.  **Commenting and Docblocks**:
    - **Docstrings**: Ensure every module, class, and function has a clear docstring summarizing its purpose, arguments, and return values (using standard conventions like Google or NumPy style).
    - **Inline Comments**: Explain *why* you are doing something, not *what* you are doing. The code itself should explain the *what* through descriptive variable and function names.
    - **Block Comments**: Use block comments to explain complex algorithms, data transformations, edge cases, or significant biological assumptions immediately preceding the relevant code block.
    - **Section Headings in Code**: Format major script sections and module definitions using 75-character dashed comment headers:
      ```python
      # ---------------------------------------------------------------------------
      # Section Title / Description
      # ---------------------------------------------------------------------------
      ```
      Use this standard structure for top-level bootstrap blocks (`# Bootstrap project root resolution for top-level imports`), project import blocks (`# Project Imports`), and constant blocks (`# Module-level Constants & Definitions`).
7.  **Duplicate Code (DRY)**: Identify duplicated logic across files or methods and consolidate it into shared utility functions or base classes.
8.  **Reusable Logic Extraction & Utility Helper Enforcement**:
    - Reusable logic belongs in `src/utils/` or `src/config/`, never re-implemented inline in analysis scripts.
    - Always use established helper functions and modules:
      - `src/utils/paths.py` -- Import directory constants (`CONFIG_DIR`, `LOG_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `PLOTS_DIR`, `REPORTS_DIR`, `SUBPROJECT_ROOT`, `PROJECT_ROOT`, `DATA_DIR`) rather than manually constructing or resolving relative file paths.
      - `src/utils/formatting.py` -- Use `generate_obsidian_frontmatter()`, `format_count_percentage()`, `format_median()`, `format_median_iqr()` for report and string formatting.
      - `src/utils/plotting.py` -- Use `save_fig()`, `resolve_colors()`.
      - `src/utils/logging.py` -- Use `TeeStream`.
      - `src/config/datasets.py` -- Use `load_dataset_config()` to load dataset metadata from `config/datasets.yaml` instead of hardcoding cohort names, paths, or dataset configurations.
      - `src/config/constants.py` & `src/biology_constants.py` -- Use central domain and pathway definitions.
      - `src/utils/dataframes.py`, `src/utils/io.py`, `src/utils/preprocessing.py` -- Use standard DataFrame, IO, and cleaning utilities.
    - Keep genuinely analysis-specific logic (e.g., `compute_cohort_frequencies`, which encodes this project's specific pathway/gene definitions) in the script itself. Don't over-extract domain logic into generic utils.
    - **Exception**: A script's very first few lines, before `sys.path` includes the project root, can't yet import `src.utils.*` (chicken-and-egg problem). Keep a small inline project-root search loop for that one bootstrap step only; everything after that should import from `src/utils/`.
9.  **Never Hardcode Values That Exist in Data, Configs, or Utilities**:
    - Never hardcode file/directory paths, dataset configurations, cohort names, or metadata values when they can be retrieved directly from `src/utils/paths.py`, `src/config/datasets.py` (`load_dataset_config()`, `datasets.yaml`), `src/config/constants.py`, or `src/biology_constants.py`.
    - Always use the helper functions established in `src/utils/` instead of re-writing custom path manipulation, YAML frontmatter formatting, figure saving, or log handling.
    - **MANDATORY FOR MARKDOWN REPORTS**: Every single reported number (sample size $N$, percentage, response rate, median, IQR, p-value, AUC, feature count, cluster count) appearing in generated markdown reports (headers, callouts, body text, tables, bullet points, key takeaways) MUST be computed on the fly from the live DataFrame/evaluation objects at runtime. NEVER hardcode numeric literals in report generation strings or templates.
    - This applies to:
      - **Cohort mutation/response frequencies**: Compute from `data/processed/<cohort>/mutations_cleaned.csv` and `clin_cleaned.csv`.
      - **Sample counts (N)**: Use `len(df_clin)`, `len(df)`, or `len(df_valid)`—NEVER bake a literal patient count integer into a string, plot title, subplot header, legend label, axis text, or markdown heading (e.g. don't write `"Liu 2019 (N=104)"` or `"Overall Survival (N=699)"` as fixed literals; construct them dynamically as `f"Liu 2019 (N={len(df_clin)})"` and `f"Overall Survival (N={len(df_valid)})"`).
      - **Markdown Text & Plot Titles/Legends/Axes**: Every patient count or sample size $N$ appearing in generated markdown reports (body text, paragraphs, bullet points, headers, callout boxes, tables, figure captions) and plot elements (titles, subplot headers, legends, axis labels, text annotations) MUST be generated dynamically from the underlying DataFrame or evaluation variable at runtime.
      - **Pooled/aggregate statistics**: Compute as a proper weighted combination of the live per-cohort values (e.g., N-weighted average), not a separate hardcoded number that can silently drift out of sync.
    - When a value genuinely can't be derived from data in this repo (e.g., no MAF file exists for a cohort), don't paper over it with a plausible-looking literal. Either:
      - Raise a clear `FileNotFoundError` explaining what's missing and how to generate it.
      - If it must be a manually-sourced value, put it in an external, clearly-labeled `data/config` file with a `source` field citing where it came from—never bury it inline in plotting code.
    - **Distinction**: This is distinct from layout/geometry constants (axis padding factors, bar widths, figure sizes). Those are legitimate to hardcode if they're derived from the data's shape where possible (e.g., category boundary lines computed from `groupby("Category").size()`, not typed as `axhline(2.5)`), and commented with why the number is what it is.
10. **Log File Location & Relative Path Logging**: Every runnable script logs its console output via `TeeStream`. Follow the pattern established in `run_pipeline.py`, with logs written to a dedicated `logs/` folder at the project root (not scattered loose files in `BASE_DIR`). When logging or printing output file and directory paths in console messages, always format them as relative paths (e.g. `path.relative_to(BASE_DIR).as_posix()`) rather than raw absolute paths:
    ```python
    from src.utils.logging import TeeStream
    import contextlib, sys

    LOG_DIR = BASE_DIR / "logs"
    LOG_PATH = LOG_DIR / "<script_name>.log"

    if __name__ == "__main__":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            stdout_tee = TeeStream(sys.stdout, log_file)
            stderr_tee = TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
                main()
    ```
    Name the log file after the script (e.g. `extended_pathway_mutation_frequencies.log`). Note: `run_pipeline.py` currently still writes `q1_pipeline.log` to the project root as a pre-existing exception -- migrate it to `logs/` next time that file is touched, for consistency.
11. **Domain Constants (`src/biology_constants.py`)**: Two kinds of "constants" get scattered through scripts, and only one kind should be centralized:
    - **Centralize** (project-wide facts that must have one source of truth):
      - Mutation classification lists (e.g., `NON_SILENT` variant classes) — these appear near-verbatim in multiple scripts; if one copy gets updated and others don't, results silently diverge between analyses.
      - Gene panel / pathway definitions (driver genes, IFN-gamma signaling genes, antigen-presentation genes, etc.) — these encode actual biological decisions. Redefining "what counts as the IFN-gamma pathway" in more than one place risks two scripts quietly disagreeing.
      - Cohort name -> directory mappings (`COHORT_DIRS`) and any other cohort-identity canonicalization repeated across data-loading scripts.
      *Put these in `src/biology_constants.py`, imported from there by any script that needs them — the same way `src/styles.py` is already the single source of truth for color/visual constants.*
    - **Keep local to the script** (presentation/script-specific choices, not project facts):
      - Row/column labels and display ordering for one specific report or figure (e.g., `ROW_LABELS`, `ROW_ORDER`).
      - Output subfolder names specific to one script's artifacts.
      - Layout/geometry constants (bar widths, padding factors) — see Rule 9 on hardcoding: legitimate if data-derived and commented, but these are inherently per-chart, not project-wide.
    - **Migration approach**: When touching an existing script and you notice it re-defines something that belongs in `biology_constants.py`, migrate it at that point rather than doing a one-off sweep across the whole repo. This keeps each change reviewable and tied to a reason for touching that file.
12. **Colors and Styling (`src/styles.py`)**:
    - Use `get_cohort_color(label)` (substring match) to resolve a cohort's color, never a direct `COHORT_PALETTE[label]` lookup — labels often carry a live-computed `(N=...)` suffix that won't equal a dict key exactly.
    - `COHORT_PALETTE` keys should be base names only (e.g., `"Liu 2019"`, not `"Liu 2019 (N=104)"`). N-suffixed keys go stale the moment a cohort's sample size changes on a rerun — don't add them back.
    - Fall back to a generated palette (e.g., `sns.color_palette(...)`) only for labels genuinely absent from `styles.py`, never a hardcoded hex value inline in a plotting script.
    - Call `set_presentation_style()` once at module load in any script that produces figures, rather than setting `plt.rcParams` locally.
13. **Functions and Closures**: Any block of logic used more than once, or that has its own clear single responsibility distinct from its enclosing function, should be its own named function — not a nested closure, not inlined. Nested closures are acceptable only when they capture enclosing-scope variables essential to their one-time use and aren't reused elsewhere.
14. **Error Handling**: Avoid bare `except:` clauses. Always catch specific exceptions (e.g., `except KeyError:`) and provide actionable error messages that explain how the user can fix the issue (e.g., `raise FileNotFoundError("Missing clin_cleaned.csv in data/processed/. Run preprocessing script first.")`).
15. **Graduate Student Report Style & Structure**: All generated markdown reports must be pitched at a graduate student level (rigorous, educational, explaining statistical and biological concepts). Every generated markdown report MUST:
    - **Include Standard Obsidian YAML Frontmatter**: Every markdown report file MUST begin with a standard Obsidian-compliant YAML frontmatter block (enclosed in `---`) generated via `generate_obsidian_frontmatter()` (from `src.utils.formatting` or local `reporting` module). Required fields include `title`, `aliases`, `tags`, `created`, `cssclasses` (`table-small`, `table-center`, `row-alt`), `obsidianEditingMode` (`preview`), `obsidianUIMode` (`source`), and `updated`.
    - **Include Standard Obsidian Callouts**: Every major report section MUST contain an Obsidian callout box (`> [!NOTE]` or `> [!INFO]`) explicitly detailing:
      1. **What is being done**
      2. **Why we are doing it**
      3. **What question it answers**
    - **Include Key Takeaways & Key Insights**: Contain a **Key Takeaways** or **Key Insights** subsection summarizing the core scientific/clinical insights. All Key Takeaways / Key Insights sections MUST be placed in `> [!INSIGHT]` callout boxes (never `> [!IMPORTANT]`).

