"""Central Domain & Biological Constants for Melanoma Analysis.

Defines canonical gene panels, mutation classification lists, activating variant sets,
and clinical metadata column exclusion lists used across subprojects.
"""

from typing import List, Set

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# The 12 pathway genes used for the BRAF/MEK/ERK ODE simulation & checkpoint axis
MODEL_GENES: List[str] = [
    "BRAF",     # RAF kinase
    "MAP2K1",   # MEK1
    "MAP2K2",   # MEK2
    "MAPK1",    # ERK2
    "MAPK3",    # ERK1
    "CDKN2A",   # p16 tumour suppressor
    "MKI67",    # proliferation marker
    "CD8A",     # cytotoxic T-cell infiltration
    "PRF1",     # perforin (immune effector)
    "GZMA",     # granzyme A (immune effector)
    "PDCD1",    # PD-1 receptor (anti-PD-1 target)
    "CD274",    # PD-L1 ligand
]

# Unique genes required for IMPRES and PD-L1 checkpoint signatures
SIGNATURE_GENES: List[str] = sorted([
    "CD274", "VSIR", "CD28", "CD276", "CD86", "TNFRSF4", "CD200", "CTLA4",
    "PDCD1", "CD80", "TNFSF9", "HAVCR2", "CD27", "CD40", "TNFRSF14",
])

# Activating BRAF mutations that confer sensitivity to BRAF inhibitors
BRAF_ACTIVATING: Set[str] = {
    "p.V600E", "p.V600K", "p.V600R", "p.V600D", "p.K601E"
}

# Activating NRAS mutations that drive baseline oncogenic MAPK signaling
NRAS_ACTIVATING: Set[str] = {
    "p.Q61R", "p.Q61K", "p.Q61L", "p.Q61H",
    "p.G12D", "p.G12R", "p.G12C", "p.G13R", "p.G13D"
}

# Key immune microenvironment features for feature screening and interaction modeling
KEY_IMMUNE_FEATURES: List[str] = [
    "TIS",
    "CYT",
    "IFN_gamma",
    "CD8_T_cells",
    "B_cells",
    "M1_M2_Ratio",
    "Macrophage_STV_Score",
]

# Macrophage polarization marker gene sets
M1_MACROPHAGE_GENES: List[str] = ["NOS2", "TNF", "IL1B", "CD68", "FCGR3A"]
M2_MACROPHAGE_GENES: List[str] = ["CD163", "MSR1", "MRC1", "CSF1R", "TGFB1"]

# Primary driver mutation columns for genomic interaction modeling
KEY_DRIVER_MUTATIONS: List[str] = [
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
]

# Metadata and clinical identifier columns to exclude from numeric feature association screening
CLINICAL_EXCLUDE_COLUMNS: List[str] = [
    "RESPONSE_BINARY",
    "PATIENT_ID",
    "SAMPLE_ID",
    "OS_STATUS",
    "OS_MONTHS",
    "RESPONSE",
    "DATASET",
    "COHORT",
    "IMMUNOTHERAPY",
    "AGE",
    "RACE",
    "SEX",
    "SPECIMEN_TYPE",
]
