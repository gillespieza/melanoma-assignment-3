"""
Unsupervised Hierarchical Clustering Analysis of Immune Signatures.

Performs batch correction on transcriptomic signatures across immunotherapy trial cohorts
(Liu 2019, Hugo 2016, Riaz 2017) using PyComBat, fits Ward hierarchical clustering,
generates a metadata-annotated clustermap, and tests cluster association with response via Chi-Square test.
"""

import contextlib
from pathlib import Path
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pycombat import Combat
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "biomarkers"
LOG_PATH = LOG_DIR / "run_clustering.log"


def _prepare_batch_corrected_signatures(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Loads cohort data, extracts signatures, and applies PyComBat batch correction.

    Args:
        data_dir: Path to project data directory.

    Returns:
        Tuple of (batch-corrected signatures DataFrame, combined response series, batch labels list).
    """
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)

    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])

    resp_col_l = "RESPONSE_BINARY" if "RESPONSE_BINARY" in clin_liu.columns else "response"
    resp_col_h = "RESPONSE_BINARY" if "RESPONSE_BINARY" in clin_hugo.columns else "response"
    resp_col_r = "RESPONSE_BINARY" if "RESPONSE_BINARY" in clin_riaz.columns else "response"

    y_liu = clin_liu.loc[sig_liu.index, resp_col_l]
    y_hugo = clin_hugo.loc[sig_hugo.index, resp_col_h]
    y_riaz = clin_riaz.loc[sig_riaz.index, resp_col_r]

    sig_all = pd.concat([sig_liu, sig_hugo, sig_riaz], axis=0)
    y_all = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
    batches = (["liu"] * len(sig_liu)) + (["hugo"] * len(sig_hugo)) + (["riaz"] * len(sig_riaz))

    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)

    print(f"Corrected signatures matrix shape: {sig_corrected.shape}")
    return sig_corrected, y_all, batches


def _evaluate_cluster_associations(sig_corrected: pd.DataFrame, y_all: pd.Series) -> None:
    """Performs agglomerative clustering into 2 groups and tests response association.

    Args:
        sig_corrected: Batch-corrected signature matrix.
        y_all: Response label series.
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

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    sig_corrected, y_all, batches = _prepare_batch_corrected_signatures(DATA_DIR)

    print("\n==================================================")
    print("Phase 2: Association of Clusters with Response...")
    print("==================================================")

    _evaluate_cluster_associations(sig_corrected, y_all)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
