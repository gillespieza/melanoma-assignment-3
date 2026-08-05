from typing import Dict, List, Set

import numpy as np

# ---------------------------------------------------------------------------
# RECIST & Clinical Outcome Mapping
# ---------------------------------------------------------------------------

RECIST_RESPONSE_MAP: Dict[str, float] = {
    "Complete Response": 1.0,
    "Partial Response": 1.0,
    "Progressive Disease": 0.0,
    "Stable Disease": np.nan,
    "Mixed Response": np.nan,
}

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# Driver genes to evaluate for somatic mutations across cohorts
DRIVER_GENES: List[str] = ["BRAF", "NRAS", "NF1"]

# Variant classifications considered non-silent/protein-altering
NON_SILENT_VARIANT_CLASSIFICATIONS: Set[str] = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}

# Specific sub-types of predicted neoantigens tracked across trial datasets
NEOANTIGEN_COLS: List[str] = [
    "SNV_NEOANTIGEN",
    "INDEL_NEOANTIGEN",
    "FUSION_NEOANTIGEN",
    "SPLICE_NEOANTIGEN",
    "VIRUS_NEOANTIGEN",
    "ERV_NEOANTIGEN",
]

# Definitions of genes included in specific functional pathways and immune signatures
PATHWAY_GENES: Dict[str, List[str]] = {
    "IFN-gamma Signature": [
        "IFNG", 
        "CXCL9", 
        "CXCL10", 
        "IDO1", 
        "HLA-DRA", 
        "STAT1"
    ],
    "Tumour Inflammation Signature (TIS)": [
        "CCL5", "CD2", "CD3D", "CD3E", "CD27", "CD274", "CMKLR1", "CXCL9",
        "CXCR6", "GZMB", "GZMK", "HLA-DRA", "HLA-DQA1", "HLA-E", "IDO1",
        "LAG3", "NKG7", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT",
    ],
    "Cytolytic Activity (CYT)": [
        "GZMA", 
        "PRF1"
    ],
    "CD8 T-Cell Abundance": [
        "CD8A", 
        "CD8B"
    ],
    "Immune Predictive Score (IMPRES)": [
        "CD274", "VSIR", "CD28", "CD276", "CD86", "TNFRSF4", "CD200",
        "CTLA4", "PDCD1", "CD80", "TNFSF9", "HAVCR2", "CD27", "CD40", "TNFRSF14",
    ],
    "Antigen Presentation": [
        "B2M", 
        "TAP1", 
        "TAP2"
    ],
    "Survival & Proliferation Drivers": [
        "PTEN", 
        "CDKN2A", 
        "PIK3CA"
    ],
}

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

