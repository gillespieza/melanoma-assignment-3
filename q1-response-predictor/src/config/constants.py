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

NEOANTIGEN_FEATURES = [
    "TMB_NONSYNONYMOUS",
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

# Definitions of genes included in specific functional pathways and immune signatures
PATHWAY_GENES = {
    "IFN-gamma Signature": ["IFNG", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "STAT1"],
    "Tumour Inflammation Signature (TIS)": [
        "CCL5", "CD2", "CD3D", "CD3E", "CD27", "CD274", "CMKLR1", "CXCL9",
        "CXCR6", "GZMB", "GZMK", "HLA-DRA", "HLA-DQA1", "HLA-E", "IDO1",
        "LAG3", "NKG7", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT",
    ],
    "Cytolytic Activity (CYT)": ["GZMA", "PRF1"],
    "CD8 T-Cell Abundance": ["CD8A", "CD8B"],
    "Immune Predictive Score (IMPRES)": [
        "CD274", "VSIR", "CD28", "CD276", "CD86", "TNFRSF4", "CD200",
        "CTLA4", "PDCD1", "CD80", "TNFSF9", "HAVCR2", "CD27", "CD40", "TNFRSF14",
    ],
    "Antigen Presentation": ["B2M", "TAP1", "TAP2"],
    "Survival & Proliferation Drivers": ["PTEN", "CDKN2A", "PIK3CA"],
}

# ---------------------------------------------------------------------------
# Immunological Signatures & Descriptive Display Names
# ---------------------------------------------------------------------------

IMMUNE_SIGNATURES = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "IMPRES",
    "PD_L1",
]

IMMUNE_SIGNATURE_LABELS = {
    "IFN_gamma": "Interferon-Gamma (IFN-γ) 6-Gene Signature",
    "TIS": "Tumour Inflammation Signature (TIS)",
    "CYT": "Cytolytic Activity (CYT) Score",
    "CD8_Tcell": "CD8 T-Cell Abundance Signature",
    "IMPRES": "Immune Predictive Score (IMPRES)",
    "PD_L1": "PD-L1 Transcript Proxy",
}

IMMUNE_SIGNATURE_GENES = {
    "IFN_gamma": ["IFNG", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "STAT1"],
    "TIS": [
        "CCL5", "CD2", "CD3D", "CD3E", "CD27", "CD274", "CMKLR1", "CXCL9",
        "CXCR6", "GZMB", "GZMK", "HLA-DRA", "HLA-DQA1", "HLA-E", "IDO1",
        "LAG3", "NKG7", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT",
    ],
    "CYT": ["GZMA", "PRF1"],
    "CD8_Tcell": ["CD8A", "CD8B"],
    "IMPRES": [
        "CD274", "VSIR", "CD28", "CD276", "CD86", "TNFRSF4", "CD200",
        "CTLA4", "PDCD1", "CD80", "TNFSF9", "HAVCR2", "CD27", "CD40", "TNFRSF14",
    ],
    "PD_L1": ["CD274"],
}

# 15 gene pairs of IMPRES signature (Auslander et al., 2018)
# Format: (Gene_A, Gene_B). Score += 1 if Gene_A > Gene_B
IMPRES_PAIRS = [
    ("CD274", "VSIR"),      # VSIR is also known as C10orf54 or VISTA
    ("CD28", "CD276"),
    ("CD86", "TNFRSF4"),
    ("CD86", "CD200"),
    ("CTLA4", "TNFRSF4"),
    ("PDCD1", "TNFRSF4"),
    ("CD80", "TNFSF9"),
    ("CD86", "HAVCR2"),
    ("CD28", "CD86"),
    ("CD27", "PDCD1"),
    ("CD40", "CD274"),
    ("CD40", "CD80"),
    ("CD40", "CD28"),
    ("CD40", "PDCD1"),
    ("TNFRSF14", "CD86"),
]


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


# ---------------------------------------------------------------------------
# Excluded Clinical & Administrative Columns
# ---------------------------------------------------------------------------

EXCLUDED_CLINICAL_COLS = [
    # Identifier columns
    "PATIENT_ID",
    "SAMPLE_ID",
    "OTHER_PATIENT_ID",
    # Administrative & timeline columns
    "DAYS_LAST_FOLLOWUP",
    "PERSON_NEOPLASM_CANCER_STATUS",
    "FORM_COMPLETION_DATE",
    "DAYS_TO_BIRTH",
    "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
    "AJCC_STAGING_EDITION",
    "INFORMED_CONSENT_VERIFIED",
    "IN_PANCANPATHWAYS_FREEZE",
    # Tissue collection & technical repository metadata
    "TISSUE_SOURCE_SITE",
    "TISSUE_SOURCE_SITE_CODE",
    "TISSUE_PROSPECTIVE_COLLECTION_INDICATOR",
    "TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR",
    "SOMATIC_STATUS",
    "MSI_SCORE_MANTIS",
    "MSI_SENSOR_SCORE",
    "TBL_SCORE",
    "SAMPLE_TYPE",
    "SAMPLE_TYPE_ID",
    # Redundant constant cancer/cohort labels
    "CANCER_TYPE",
    "CANCER_TYPE_ACRONYM",
    "CANCER_TYPE_DETAILED",
    "TUMOR_TYPE",
    "SUBTYPE",
    "ONCOTREE_CODE",
    # Post-baseline treatment/event & presentation assessment columns
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
    "PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT",
]