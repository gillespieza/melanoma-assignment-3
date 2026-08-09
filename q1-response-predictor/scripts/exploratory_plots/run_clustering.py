"""
Unsupervised Hierarchical Clustering Analysis of Immune Signatures.

Performs batch correction on transcriptomic signatures across all active
immunotherapy trial cohorts (loaded dynamically from config/datasets.yaml)
using PyComBat, fits Ward hierarchical clustering, and tests cluster
association with response via Chi-Square test.
"""

import contextlib
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pycombat import Combat
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.append(str(_SUBPROJECT_ROOT))

from src.data_loaders import load_all_active_cohorts
from src.signatures import extract_all_signatures
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

_CONFIG_PATH = _SUBPROJECT_ROOT / "config" / "datasets.yaml"

# Response column names accepted from cleaned clinical data, in priority order.
_RESP_COLS = ("RESPONSE_BINARY", "response")

_LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = _LOG_DIR / "run_clustering.log"


def _prepare_batch_corrected_signatures(
    data_dir: Path,
    config_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Loads all active trial cohorts dynamically, extracts immune signatures,
    and applies PyComBat batch correction.

    Cohorts are discovered from config/datasets.yaml via load_all_active_cohorts.
    Cohorts lacking a valid response label (RESPONSE_BINARY or response) are
    skipped with a warning rather than raising an error.

    Args:
        data_dir: Path to project data directory.
        config_path: Path to datasets.yaml configuration file.

    Returns:
        Tuple of (batch-corrected signatures DataFrame, combined response series,
        batch label list).

    Raises:
        RuntimeError: If fewer than two cohorts have valid response data (batch
            correction is undefined with a single batch).
    """
    expr_dict, clin_dict, _, trial_names = load_all_active_cohorts(
        config_path=config_path,
        data_dir=data_dir,
        merge_only=True,
    )

    # Restrict to trial cohorts that have expression data loaded.
    trial_expr = {name: expr_dict[name] for name in trial_names if name in expr_dict}
    if not trial_expr:
        raise RuntimeError(
            "No active trial cohorts found with expression data. "
            "Check datasets.yaml and that processed CSVs exist on disk."
        )

    # Compute intersection of genes across all trial cohorts.
    common_genes = None
    for expr in trial_expr.values():
        common_genes = (
            expr.columns if common_genes is None
            else common_genes.intersection(expr.columns)
        )

    print(f"Common genes across {len(trial_expr)} trial cohort(s): {len(common_genes)}")

    sig_parts: List[pd.DataFrame] = []
    resp_parts: List[pd.Series] = []
    batch_labels: List[str] = []

    for name in trial_names:
        if name not in trial_expr:
            continue

        clin = clin_dict[name]
        expr = trial_expr[name]

        resp_col = next((c for c in _RESP_COLS if c in clin.columns), None)
        if resp_col is None:
            print(
                f"  [SKIP] {name}: no response column found "
                f"({', '.join(_RESP_COLS)})."
            )
            continue

        y = clin.loc[expr.index, resp_col].dropna()
        if y.empty:
            print(f"  [SKIP] {name}: response column is all-NaN after dropping missing values.")
            continue

        expr_aligned = expr.loc[y.index, common_genes]
        sig = extract_all_signatures(expr_aligned)

        sig_parts.append(sig)
        resp_parts.append(y.loc[sig.index])
        batch_labels.extend([name] * len(sig))
        print(f"  Loaded {name}: {len(sig)} samples")

    if len(sig_parts) < 2:
        raise RuntimeError(
            f"Batch correction requires at least 2 cohorts with valid response data; "
            f"only {len(sig_parts)} qualified. Ensure RESPONSE_BINARY is present in "
            "cleaned clinical files for the relevant trial cohorts."
        )

    sig_all = pd.concat(sig_parts, axis=0)
    y_all = pd.concat(resp_parts, axis=0)

    sig_corrected_arr = Combat().fit_transform(sig_all.values, batch_labels)
    sig_corrected = pd.DataFrame(
        sig_corrected_arr, index=sig_all.index, columns=sig_all.columns
    )

    print(f"Corrected signatures matrix shape: {sig_corrected.shape}")
    return sig_corrected, y_all, batch_labels


def _evaluate_cluster_associations(
    sig_corrected: pd.DataFrame,
    y_all: pd.Series,
) -> None:
    """Performs agglomerative clustering into 2 groups and tests response association.

    Args:
        sig_corrected: Batch-corrected signature matrix.
        y_all: Response label series aligned to sig_corrected index.
    """
    cluster_model = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
    patient_clusters = cluster_model.fit_predict(sig_corrected)

    df_cluster_assoc = pd.DataFrame({"Cluster": patient_clusters, "Response": y_all})

    contingency_table = pd.crosstab(df_cluster_assoc["Cluster"], df_cluster_assoc["Response"])
    print("\nPatient Cluster vs. Immunotherapy Response Contingency Table:")
    print(contingency_table)

    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    print("\nChi-Square Test statistics:")
    print(f"  Chi-Square: {chi2:.3f}")
    print(f"  p-value: {p_val:.3e}")

    for cluster in [0, 1]:
        cluster_data = df_cluster_assoc[df_cluster_assoc["Cluster"] == cluster]
        resp_rate = cluster_data["Response"].mean() * 100
        print(f"  Cluster {cluster} Response Rate: {resp_rate:.1f}% (N={len(cluster_data)})")


def main() -> None:
    """Executes unsupervised hierarchical clustering workflow."""
    print("==================================================")
    print("Phase 1: Loading & Batch-Correcting Cohort Signatures...")
    print("==================================================")

    sig_corrected, y_all, _ = _prepare_batch_corrected_signatures(DATA_DIR, _CONFIG_PATH)

    print("\n==================================================")
    print("Phase 2: Association of Clusters with Response...")
    print("==================================================")

    _evaluate_cluster_associations(sig_corrected, y_all)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()
