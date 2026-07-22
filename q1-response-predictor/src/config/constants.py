"""Domain constants used by the Q1 melanoma analysis pipeline."""

import numpy as np


# ---------------------------------------------------------------------------
# Clinical response
# ---------------------------------------------------------------------------

RECIST_RESPONSE_MAP = {
    "Complete Response": 1,
    "Partial Response": 1,
    "Progressive Disease": 0,
    "Stable Disease": np.nan,
    "Mixed Response": np.nan,
}

# ---------------------------------------------------------------------------
# Dataset Support
# ---------------------------------------------------------------------------

SUPPORTED_PROCESSING_STRATEGIES = (
    "iatlas",
    "tcga",
)

# ---------------------------------------------------------------------------
# Protected biological genes
# ---------------------------------------------------------------------------

# Variant classifications considered non-silent/protein-altering
NON_SILENT_VARIANT_CLASSIFICATIONS = {
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

GENOMIC_FEATURES = [
    "TMB_NONSYNONYMOUS",
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
    "SNV_NEOANTIGEN",
    "INDEL_NEOANTIGEN",
    "FUSION_NEOANTIGEN",
    "SPLICE_NEOANTIGEN",
    "CTA_SELF_NEOANTIGEN",
]

# Driver genes to evaluate for mutations
DRIVER_GENES = ["BRAF", "NRAS", "NF1"]

# Extended driver genes for the single-cohort and merged CoMut plots
COMUT_DRIVER_GENES = ['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'PTEN', 'JAK1', 'JAK2', 'B2M', 'TAP1', 'TAP2']
MERGED_COMUT_DRIVER_GENES = ['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'PTEN', 'KIT', 'TP53', 'JAK1', 'JAK2', 'B2M']

# Signature genes for the CoMut plot
SIGNATURE_GENES = [
    'GBP4', 'CCL8', 'GBP5', 'KLRD1', 'PLAAT4', 'GPR171', 'IDO1',
    'CXCL11', 'CXCL10', 'PTPN22', 'GBP1', 'CD72', 'STAT4', 'IL15',
    'AKAP5', 'SAMSN1', 'GBP1P1', 'ZNF831', 'KLRK1', 'CD38'
]

# Definitions of genes included in specific functional pathways
PATHWAY_GENES = {
    "Antigen Presentation": ["B2M", "TAP1", "TAP2"],
    "IFN-gamma Signaling": ["JAK1", "JAK2", "STAT1"],
    "Survival & Proliferation Drivers": ["PTEN", "CDKN2A", "PIK3CA"],
}


PROTECTED_GENES = {
    # IMPRES genes
    "CD274",
    "VSIR",
    "C10orf54",
    "VISTA",
    "CD28",
    "CD276",
    "CD86",
    "TNFRSF4",
    "CD200",
    "CTLA4",
    "PDCD1",
    "CD80",
    "TNFSF9",
    "HAVCR2",
    "CD27",
    "CD40",
    "TNFRSF14",

    # IFN-gamma genes
    "IFNG",
    "CXCL9",
    "CXCL10",
    "IDO1",
    "HLA-DRA",
    "STAT1",

    # T-cell inflamed signature genes
    "CCL5",
    "CD2",
    "CD3D",
    "CD3E",
    "CMKLR1",
    "CXCR6",
    "GZMB",
    "GZMK",
    "HLA-DQA1",
    "HLA-E",
    "LAG3",
    "NKG7",
    "PDCD1LG2",
    "PSMB10",
    "TIGIT",

    # Cytolytic activity genes
    "GZMA",
    "PRF1",

    # CD8 T-cell genes
    "CD8A",
    "CD8B",

    # Antigen presentation and signalling
    "B2M",
    "TAP1",
    "TAP2",
    "JAK1",
    "JAK2",

    # Melanoma driver and pathway genes
    "PTEN",
    "CDKN2A",
    "PIK3CA",
    "BRAF",
    "NRAS",
    "NF1",
}