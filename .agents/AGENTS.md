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
- **Kaplan-Meier Plotting Standards**: All Kaplan-Meier (KM) survival plots generated across the codebase MUST:
  1. Always include a **numbers-at-risk table** beneath the plot axis via `add_km_risk_table(kmf_list, ax)` (from `src.utils.plotting`), providing clinical transparency on sample size attrition over time.
  2. Always display **confidence interval shading** (`ci_show=True`, `ci_alpha=0.12` or `alpha=0.15`) on survival curves, unless explicitly instructed otherwise by the user.
- **Python Imports**: Import central color palettes from `src.styles` (`from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style`).

---

## 3. Language & Formatting Conventions

- **Spelling**: Always use **British English** spelling for all human-readable output text (e.g., *colour*, *visualisation*, *characterisation*, *tumour*, *analyse*, *modelling*, *centre*). The scope of this rule is:
  - ✅ **In scope** — apply British spelling to: markdown report body text and headings; code **comments** (`# ...`); **docstrings** (`"""..."""`); **plot title and axis label strings** (e.g. `set_title(...)`, `set_xlabel(...)`, `ax.text(...)`); **print/log output strings** and `f.write(...)` report content.
  - ❌ **Out of scope** — do **not** apply British spelling to: Python **function argument names or kwarg values** (e.g. `ha="center"`, `va="center"`, `loc="lower center"`, `color=...`, `edgecolor=...` — these are library API tokens and must match exactly what the library expects); **YAML key strings** in frontmatter or cssclass lists (e.g. `table-center` is the Obsidian CSS class name and must not be changed); **Python dictionary key strings** used as internal code identifiers (e.g. `{"color": "#aabbcc"}`); **import lines and module/class names** (e.g. `from matplotlib.colors import ...`); **variable names, loop variables, and function parameter names** (e.g. `for color in palette`, `return color`).
- **Gene Names**: Always wrap all gene names and gene symbols in backticks (e.g., `CD274`, `PDCD1`, `BRAF`, `BRAF V600`, `BRAF V600E`, `NRAS`, `NF1`, `B2M`, `TAP1`, `JAK1`, `STAT1`) across all generated markdown files, documentation, reports, and text artifacts.
- **Canonical Immune Signature Order**: Whenever immune signatures are listed — in code (`List[str]` constants, function arguments, column selectors), report body text, table column headers, plot axis tick labels, radar spokes, legend entries, or feature importance tables — they MUST always appear in this fixed order:
  1. `IFN_gamma` (IFN-γ Signature)
  2. `TIS` (Tumour Inflammation Score)
  3. `CYT` (Cytolytic Activity Score)
  4. `CD8_Tcell` (CD8+ T-cell Abundance)
  5. `IMPRES` (Immune Predictive Score)
  6. `PD_L1` (PD-L1 Expression Proxy)

  This order is biologically motivated: it groups T-cell activation signatures together before moving to the checkpoint/exhaustion axis. Any re-ordering (e.g. alphabetical or arbitrary) is a style error. This applies project-wide across all scripts, notebooks, and generated artefacts.
- **Canonical ML Model Presentation Order**: Whenever the five ML classifiers are listed — in report tables, heatmap row/column labels, legend entries, code loops, or any other enumeration — they MUST always appear in this fixed order:
  1. XGBoost (XGB)
  2. Random Forest (RF)
  3. Support Vector Machine (SVM)
  4. Elastic-Net
  5. Logistic Regression (LR)

  This order is performance-motivated: it leads with the strongest tree-based models, then the kernel method, then regularised linear models, finishing with the baseline linear classifier. Any re-ordering is a style error. This applies project-wide across all scripts, notebooks, and generated artefacts.
- **Horizontal Rules**: Never use horizontal rules (`---` or `***`) in markdown files or responses unless explicitly instructed to do so (the project's custom CSS automatically renders horizontal rules beneath `<h1>` and `<h2>` headings).


---

## 4. Code Smells & Refactoring Guidelines

When fixing code smells or refactoring code in this repository, follow these guidelines:

1.  **Unused Imports & Variables**: Remove all unused imports and variables. Ensure your imports are organized according to standard PEP 8 conventions.
2.  **Long Functions**: Break down functions longer than 30 lines into smaller, well-named helper functions that do one thing well (Single Responsibility Principle).
3.  **Complex Conditionals**: Extract complex nested `if/else` logic into well-named boolean variables or separate evaluation functions to improve readability.
3b. **Dead Code (No-Ops)**: Remove any block that has no effect — e.g., `if condition: pass`, `result = result`, `x = x`. These are frequently left behind after iterative edits and silently mislead readers into thinking the branch has a purpose.
4.  **Magic Numbers/Strings**: Replace hard-coded values and magic numbers with descriptive, uppercase module-level constants (e.g., `MAX_ITERATIONS = 100`). For schema column name strings local to a single script, use private `_COL_*` module-level constants (e.g., `_COL_SAMPLE_ID = "SAMPLE_ID"`) rather than repeating bare string literals throughout the file.
4b. **Line Length**: Keep all lines to a maximum of **100 characters**. Break long Pandas chains across multiple lines using implicit line continuation inside brackets. This is stricter than PEP 8 (79) but more readable than no limit; it is a firm project-wide standard.
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
      - `src/utils/paths.py` -- Import directory constants (`LOG_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `PLOTS_DIR`, `REPORTS_DIR`, `PROJECT_ROOT`, `DATA_DIR`) rather than manually constructing or resolving relative file paths. **⚠ Do NOT import `CONFIG_DIR`** — it does not exist in `paths.py` and will raise an `ImportError` at runtime. **⚠ Avoid `SUBPROJECT_ROOT`** for subproject-local config paths — because `paths.py` lives in root-level `src/`, `SUBPROJECT_ROOT` evaluates relative to that file and resolves to the project root, not the calling subproject. For configs local to a subproject, derive the path from `__file__` instead:
        ```python
        # Resolves correctly regardless of CWD
        _SCRIPT_DIR = Path(__file__).resolve().parent
        CONFIG_PATH = _SCRIPT_DIR.parent / "config" / "datasets.yaml"
        ```
      - `src/utils/execution.py` -- Use `run_companion_scripts()` for executing companion plot/analysis scripts in isolated subprocesses.
      - `src/utils/formatting.py` -- Use `generate_obsidian_frontmatter()`, `format_count_percentage()`, `format_median()`, `format_median_iqr()` for report and string formatting.
      - `src/utils/plotting.py` -- Use `save_fig()`, `resolve_colors()`.
      - `src/utils/logging.py` -- Use `TeeStream`.
      - `src/config/datasets.py` -- Use `load_dataset_config()` to load dataset metadata from `config/datasets.yaml` instead of hardcoding cohort names, paths, or dataset configurations.
      - `src/config/constants.py` & `src/biology_constants.py` -- Use central domain and pathway definitions.
      - `src/utils/dataframes.py`, `src/utils/io.py`, `src/utils/preprocessing.py` -- Use standard DataFrame, IO, and cleaning utilities.
    - Keep genuinely analysis-specific logic (e.g., `compute_cohort_frequencies`, which encodes this project's specific pathway/gene definitions) in the script itself. Don't over-extract domain logic into generic utils.
    - **Exception**: A script's very first few lines, before `sys.path` includes the project root, can't yet import `src.utils.*` (chicken-and-egg problem). Keep a small inline project-root search loop for that one bootstrap step only; everything after that should import from `src/utils/`.
9.  **Never Hardcode Values That Exist in Data, Configs, or Utilities**:
    - **Dynamic Cohort & Dataset Loading (Futureproofing Mandate)**: NEVER hardcode cohort names (e.g. `["Liu 2019", "Hugo 2016", "Riaz 2017"]`) or fixed dataset lists inside analysis scripts, plotters, or report generators. All scripts MUST load active datasets dynamically from `config/datasets.yaml` via `load_dataset_config()` or scan `data/processed/` at runtime so the pipeline automatically supports newly added datasets without requiring code modifications.
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
10. **Log File Location & Relative Path Logging**: Every runnable script logs its console output via `TeeStream`.

    **Subproject scripts** (anything under `q1-response-predictor/`, `q5-patient-stratification/`, etc.) MUST write logs to their own subproject `logs/` folder — never to the top-level project `logs/`. Derive the log directory from the script's own `__file__` path, not from the imported `LOG_DIR` in `src/utils/paths.py` (which resolves to `PROJECT_ROOT / "logs"` and is therefore wrong for subproject scripts):

    ```python
    _SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]  # e.g. q1-response-predictor/
    LOG_DIR  = _SUBPROJECT_ROOT / "logs"
    LOG_PATH = LOG_DIR / "<script_name>.log"
    ```

    Do NOT import `LOG_DIR` from `src.utils.paths` in subproject scripts — that constant points to the top-level `melanoma-assignment-3/logs/` and will silently write logs to the wrong location.

    When logging or printing output file and directory paths in console messages, always format them as relative paths rather than raw absolute paths. Use `_SUBPROJECT_ROOT` as the base for all relative path display within subproject scripts:

    ```python
    from src.utils.logging import TeeStream
    import contextlib, sys

    _SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
    LOG_DIR  = _SUBPROJECT_ROOT / "logs"
    LOG_PATH = LOG_DIR / "<script_name>.log"

    if __name__ == "__main__":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            stdout_tee = TeeStream(sys.stdout, log_file)
            stderr_tee = TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
                main()
    ```

    Name the log file after the script (e.g. `run_clinical_analysis.log`). Note: `run_pipeline.py` currently still writes `q1_pipeline.log` to the project root as a pre-existing exception — migrate it to `q1-response-predictor/logs/` next time that file is touched, for consistency.
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
    - **Include Standard Obsidian YAML Frontmatter**: Every markdown report file MUST begin with a standard Obsidian-compliant YAML frontmatter block (enclosed in `---`) generated via `generate_obsidian_frontmatter()` (from `src.utils.formatting` or local `reporting` module). Required fields include `title`, `aliases`, `tags`, `created`, `cssclasses` (`table-small`, `table-center`, `row-alt`), and `updated`.
    - **Include Standard Obsidian Callouts**: Every major report section MUST contain an Obsidian callout box (`> [!NOTE]` or `> [!INFO]`) explicitly detailing:
      1. **What is being done** (Heading MUST use abbreviated: `**What**`, never `**What we are doing**` or `**What We Are Doing**`)
      2. **Why we are doing it** (Heading MUST use abbreviated: `**Why**`, never `**Why we are doing it**` or `**Why We Are Doing It**`)
      3. **What question it answers** (Heading MUST use abbreviated: `**Question(s)**`, `**Question**`, or `**Questions**`)
    - **Include Key Takeaways & Key Insights**: Contain a **Key Takeaways** or **Key Insights** subsection summarizing the core scientific/clinical insights. All Key Takeaways / Key Insights sections MUST be placed in `> [!INSIGHT]` callout boxes (never `> [!IMPORTANT]`).
    - **Student Report Limitations Callout Requirement (`phase_x_STUDENT.md`)**: All Limitations & Future Directions sections in `phase_x_STUDENT.md` report files MUST be rendered within `> [!WARNING]` callout boxes.
16. **Ground-Truth-First Report Verification Protocol**: Before writing or updating any claim in a phase report about what features, methods, or values the pipeline *uses* or *produces*, you MUST verify against live output files — not code, comments, or prior documentation. Violating this rule is what causes "excluded feature" claims to contradict the actual CSV, and count mismatches between reports and runtime logs.

    **Step 1 — Read the output file, not the code.**
    For any claim about features, columns, or sample sizes, inspect the actual output CSV directly:
    ```python
    import pandas as pd
    df = pd.read_csv("data/processed/q5/feature_matrix.csv")
    print(df.columns.tolist(), df.shape)
    ```
    Do NOT derive counts from code comments, docstrings, variable names, or prior documentation — these reflect *intent*, not *reality*.

    **Step 2 — Separate metadata from engineered features.**
    Never report `df.shape[1]` as a "feature count". Identify metadata / clinical label columns (e.g. `SAMPLE_ID`, `PATIENT_ID`, `COHORT`, outcome labels) and subtract them explicitly. Use a named constant (e.g. `METADATA_COLS`) — never a bare magic number.

    **Step 3 — Verify claimed exclusions against the file.**
    If the report will state a feature was "dropped", "excluded", or "removed":
    - Confirm it is **absent** from the output CSV column list.
    - If it **is present** in the CSV: it is *retained*, regardless of intent in the code or comments.
    - If it **is referenced** in any downstream script's feature list: document it as retained and scope-limited, not excluded. The file is the source of truth; code comments are aspirational.

    **Step 4 — Cross-check counts against runtime logs.**
    If a pipeline log exists (e.g. `logs/q5_pipeline.log`), compare its logged column/feature counts against your Step 1 direct CSV inspection. If they disagree, the CSV inspection wins — investigate the log discrepancy rather than trusting the log blindly.

    **Step 5 — Use dynamic values in all report text.**
    Every number in a markdown report (N, feature count, response rate, cluster size) MUST be computed from the live data object at report-generation time. If the report generator contains a hardcoded count (e.g. `n = 32`), replace it with a live computation (e.g. `len([c for c in df.columns if c not in METADATA_COLS])`).

    **The specific failure mode this rule prevents:**
    > ✗ *"IMPRES was excluded from the panel"* — written after reading code intent, without checking whether `IMPRES` is a column in `feature_matrix.csv`.
    > ✓ *"`IMPRES` appears in column 16 of `feature_matrix.csv`; it is retained in the multi-modal feature matrix. Its scope is limited to Phase 5/6 predictive models rather than the TME deconvolution core."*
17. **Windows & PowerShell Command Execution Safety**: When executing terminal commands on Windows (`pwsh`):
    - **Always Prefer Scratch Files Over `python -c`** *(unconditional default)*: For **any** Python snippet beyond a single expression, write the code to a scratch file first (e.g. `write_to_file` → `scratch/check_snippet.py`) and execute `python scratch/check_snippet.py`. Do **not** attempt to inline it with `python -c "..."` first and fall back to a scratch file only on failure — go straight to the scratch file every time.
    - **`python -c` Is Permitted Only for True One-Liners**: A single, quote-free expression (e.g. `python -c "import sys; print(sys.version)"`) is the only acceptable use of `python -c`. If the snippet contains any of the following, it **must** be a scratch file instead: multiple statements, f-strings, nested quotes, backslashes, `import` + logic, or more than ~60 characters.
    - **Scratch File Location**: Store ephemeral check scripts in the artifact scratch directory (`C:\Users\Amanda\.gemini\antigravity\brain\<conversation-id>\scratch\`) or in `scratch/` at the project root. Name them descriptively (e.g. `check_columns.py`, `smell_check.py`). They are auto-persisted and do not need cleanup.
    - **PowerShell Compatibility**: Ensure all shell commands use flags and syntax compatible with PowerShell on Windows (e.g. forward slashes or escaped backslashes for paths, standard pwsh cmdlets or cross-platform binaries).
    - **Explicit UTF-8 File Encoding**: Always explicitly specify `encoding="utf-8"` when reading or writing text files in Python scripts (`open(..., encoding="utf-8")`, `Path.read_text(encoding="utf-8")`, `Path.write_text(..., encoding="utf-8")`) to prevent Windows default `cp1252` `UnicodeDecodeError` failures.
    - **Console Output UTF-8 Encoding**: Reconfigure standard output encoding in scripts printing non-ASCII/Unicode characters (e.g., `sys.stdout.reconfigure(encoding="utf-8")` or `encoding="utf-8"` in log streams) to avoid `UnicodeEncodeError: 'charmap' codec can't encode character` when running on Windows.
18. **Verify Imports Against Module Exports Before Writing Them**: Before writing `from module import Name`, verify that `Name` is actually exported by the target module. Do not assume a name exists because it *should* logically exist or was mentioned in documentation. For shared utility modules (`src/utils/paths.py`, `src/utils/formatting.py`, etc.), read the file and confirm exported names before importing. This prevents `ImportError` crashes that only surface at runtime.
    - **In practice**: Read `paths.py` before importing any path constant — its export list is short and changes over time. If a constant you need is missing, either add it to the utility module (with a note in the session log) or derive it locally using `Path(__file__)`. Never invent an import name based on what *seems* like it should exist.
19. **Large Script Refactoring & Timeout Prevention Protocol**:
    - **Incremental Refactoring**: When refactoring large scripts (>500 lines) for AST function lengths or line limits, edit only 2–3 related functions per tool call. Never attempt to replace hundreds of lines across multiple non-contiguous sections in a single macro edit.
    - **AST Symbol & Dependency Verification**: Before deleting, splitting, or renaming helper functions, run a scratch AST check script (`ast.parse`) to verify all downstream references and callers are updated simultaneously so no `NameError` or dropped symbol exceptions occur.
    - **Compilation & Execution Verification**: Run `python -m py_compile <script>` after every refactoring pass to verify syntax before launching long-running pipeline tasks.
    - **Active Progress Stream Heartbeat**: Always output a brief status summary in chat before launching long background tasks or complex multi-tool sequences to prevent empty response timeouts and maintain stream heartbeats.
20. **Naming Conventions — Strictly No Hyphens in Folder or File Names**:
    - **Underscores Only**: NEVER use hyphens (`-`) in folder, package, module, subproject directory, or file names. ALWAYS use underscores (`_`) (snake_case).
    - **Python Import Compatibility**: Hyphens in directory or file names make them invalid Python identifiers, breaking standard Python imports (`from package.sub_module import ...` raises `SyntaxError`) and requiring awkward dynamic import workarounds. All directories, scripts, test files, and assets must use lowercase alphanumeric characters and underscores only.
21. **Van Allen 2015 Cohort — Permanently Excluded from This Branch**:
    - **Van Allen 2015 (`van_allen_2015`) is REMOVED from this branch and must NOT be included in any merged immunotherapy datasets, ML training sets, LOCO CV folds, feature matrices, or reports.**
    - **Reason**: Van Allen 2015 is a pure ipilimumab (aCTLA-4) cohort — all 110 patients received ipilimumab as their primary treatment. It is a fundamentally different drug class from the anti-PD-1 trials (Liu 2019, Hugo 2016, Riaz 2017, Gide 2019). Including it conflates CTLA-4 response biology with PD-1 response biology, introducing a qualitatively different confounder that cannot be corrected by covariate adjustment. The scientific decision is documented as **ASD-03** in `q1_response_predictor/docs/DECISIONS.md`.
    - **Practical enforcement**: In `config/datasets.yaml`, Van Allen 2015 must be either removed or marked `active: false`. In `merge_datasets.py` and all downstream scripts, do NOT load or reference `van_allen_2015` data. Do NOT add it back or re-enable it without an explicit instruction from Amanda.
    - **Scope**: This exclusion applies to Q1 (response predictor), Q1.1 (patient stratification), and all merged dataset operations. Van Allen 2015 data files may remain on disk in `data/raw/van_allen_2015/` and `data/processed/van_allen_2015/` for archival purposes but must not be loaded into any active pipeline.
