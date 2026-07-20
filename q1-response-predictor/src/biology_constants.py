"""Project-wide biological constants and domain definitions."""

# Canonical directory mappings for cohorts
COHORT_DIRS = {
    "Liu 2019": "liu_2019",
    "Hugo 2016": "hugo_2016",
    "Riaz 2017": "riaz_2017",
}

# Variant classifications considered non-silent/protein-altering
NON_SILENT = [
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
]

# Driver genes to evaluate for mutations
DRIVER_GENES = ["BRAF", "NRAS", "NF1"]

# Extended driver genes for the CoMut plot
COMUT_DRIVER_GENES = ['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'PTEN', 'JAK1', 'JAK2', 'B2M', 'TAP1', 'TAP2']

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

