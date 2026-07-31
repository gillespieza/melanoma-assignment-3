"""Q5 Domain Constants.

Defines feature sets, resistance pathways, treatability index parameters,
Q3 ODE simulation initial conditions, and Q4 DepMap/LINCS drug target mappings.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# Marker gene panels for transcriptomic cell-type deconvolution
CELL_TYPE_MARKERS: Dict[str, List[str]] = {
    "CD8_T_cells": ["CD8A", "CD8B", "CD3D", "CD3E"],
    "CD4_T_cells": ["CD4", "IL7R", "FOXP3"],
    "NK_cells": ["NCAM1", "KLRB1", "NCR1"],
    "B_cells": ["CD19", "MS4A1", "CD79A"],
    "M1_Macrophages": ["TNF", "IL12B", "CXCL10", "NOS2", "IRF5"],
    "M2_Macrophages": ["CD163", "MRC1", "MSR1", "TGFB1", "ARG1"],
    "CAFs": ["FAP", "PDGFRB", "COL1A1", "ACTA2"],
}

# Gene sets for core immune signatures (fallback calculations)
IMMUNE_SIGNATURE_MARKERS: Dict[str, List[str]] = {
    "TIS": ["CD274", "PDCD1", "STAT1", "HLA-DRA", "CXCL9", "CXCL10", "IDO1"],
    "CYT": ["PRF1", "GZMA"],
    "IFN_gamma": ["IFNG", "STAT1", "IDO1", "CXCL9", "CXCL10"],
    "CD8_Tcell": ["CD8A", "CD8B"],
}

# Spatial microenvironment indicators (resolving stroma exclusion vs immune desert)
SPATIAL_MICROENVIRONMENT_FEATURES: List[str] = [
    "Spatial_CD8_CAF_Distance_Ratio",
    "Spatial_Tumour_Infiltration_Index",
]

# Features utilized for patient clustering (9 primary multi-modal features)
CLUSTERING_FEATURES: List[str] = [
    "TIS",
    "CYT",
    "CD8_T_cells",
    "M1_Macrophages",
    "M2_Macrophages",
    "CAFs",
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
]

# Features utilized for phenotype profiling and label assignment (includes spatial indicators)
PHENOTYPE_PROFILE_FEATURES: List[str] = [
    "TIS",
    "CYT",
    "CD8_T_cells",
    "mut_NF1",
    "M1_M2_Ratio",
    "M2_Macrophages",
    "Spatial_CD8_CAF_Distance_Ratio",
    "Spatial_Tumour_Infiltration_Index",
]

# Descriptive biological suffixes for phenotype labels
PHENOTYPE_LABEL_SUFFIX: Dict[str, str] = {
    "Immune Hot": "(High TIS & CYT, Inflamed Microenvironment)",
    "Immune Cold": "(Low TIS & Infiltration, Desert)",
    "Mutant-Driven": "(NF1 Loss & High Response Subtype)",
    "Immunosuppressive M2-High": "(Depleted T-cells & Stromal Exclusion)",
}

# Resistance pathways and key associated genes
RESISTANCE_PATHWAYS: Dict[str, List[str]] = {
    "Antigen_Presentation_Loss": ["B2M", "TAP1", "TAP2", "HLA-A", "HLA-B", "HLA-C"],
    "IFN_Gamma_Resistance": ["JAK1", "JAK2", "STAT1", "STAT3"],
    "Immunosuppressive_Microenvironment": ["IL10", "TGFB1", "IDO1", "ARG1"],
    "High_Proliferation_Glycolysis": ["MYC", "MTOR", "LDHA", "PFKFB3"],
}

# Features evaluated within the treatability index
TREATABILITY_FEATURES: List[str] = [
    "antigen_presentation_intact",
    "ifn_gamma_signaling_intact",
    "low_cna_burden",
    "actionable_mutations_present",
]

# Standard phenotype labels
PHENOTYPE_LABELS: Dict[str, str] = {
    "HOT": "Immune Hot",
    "COLD": "Immune Cold",
    "SUPPRESSIVE": "Immunosuppressive M2-High",
    "MUTANT": "Mutant-Driven",
}

# Mapping from canonical phenotype display name to its Phase 3 GMM output probability column.
# Keyed by PHENOTYPE_LABELS values to guarantee label consistency across the codebase.
# Named columns (P_Immune_Hot etc.) are used instead of the raw P_Cluster_k columns because
# GMM cluster integer indices are arbitrary and can shift between runs; named columns are stable.
PHENOTYPE_PROB_COL: Dict[str, str] = {
    "Immune Hot":                "P_Immune_Hot",
    "Immune Cold":               "P_Immune_Cold",
    "Immunosuppressive M2-High": "P_Immunosuppressive_M2_High",
    "Mutant-Driven":             "P_Mutant_Driven",
}

# Q3 ODE initial parameters mapped per Q5 phenotype
Q3_ODE_PHENOTYPE_PARAMS: Dict[str, Dict[str, float]] = {
    "Immune Hot": {"T0": 1.0, "E0": 0.8, "M0": 0.2, "a_kill": 0.15},
    "Immune Cold": {"T0": 1.0, "E0": 0.1, "M0": 0.1, "a_kill": 0.02},
    "Immunosuppressive M2-High": {"T0": 1.0, "E0": 0.3, "M0": 0.8, "a_kill": 0.04},
    "Mutant-Driven": {"T0": 1.0, "E0": 0.5, "M0": 0.3, "a_kill": 0.08},
}

# Q4 DepMap essentiality targets and LINCS perturbagen recommendations per phenotype
Q4_TARGET_NOMINATIONS: Dict[str, Dict[str, List[str]]] = {
    "Immune Cold": {
        "depmap_targets": ["AXL", "MDM2", "CDK4"],
        "lincs_perturbagens": ["Vorinostat", "Entinostat"],
        "strategy": "Priming / Epigenetic Immune Sensitisation + Anti-PD-1",
    },
    "Immunosuppressive M2-High": {
        "depmap_targets": ["CSF1R", "IDO1", "ARG1"],
        "lincs_perturbagens": ["PLX3397", "Epacadostat"],
        "strategy": "M2 Macrophage Depletion / Reprogramming + Anti-PD-1",
    },
    "Mutant-Driven": {
        "depmap_targets": ["BRAF", "MAP2K1", "PIK3CA"],
        "lincs_perturbagens": ["Dabrafenib", "Trametinib"],
        "strategy": "Targeted Kinase Inhibition + Checkpoint Blockade",
    },
}
