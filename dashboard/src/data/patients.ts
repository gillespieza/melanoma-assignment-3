import type { PatientInput } from "./types";

// ---------------------------------------------------------------------------
// Three hardcoded demo patients — each a clean clinical archetype that the
// q3 ODE model separates cleanly. Numbers are representative of the TCGA-SKCM
// subgroups the model was run on.
// ---------------------------------------------------------------------------

export const PATIENTS: PatientInput[] = [
  {
    id: "A",
    displayName: "Patient A · Ms. R. Doyle",
    age: 61,
    sex: "F",
    ecog: 1,
    stage: "IVb (M1b)",
    ldh: "High (>2x ULN)",
    braf: "V600E",
    nras: "WT",
    pdl1: 3,
    signature: 22,
    vignette:
      "BRAF V600E, PD-L1 low, bulky lung metastases with high LDH. Symptomatic and needs rapid disease control.",
  },
  {
    id: "B",
    displayName: "Patient B · Mr. T. Okafor",
    age: 54,
    sex: "M",
    ecog: 0,
    stage: "IVa (M1a)",
    ldh: "Normal",
    braf: "WT",
    nras: "WT",
    pdl1: 65,
    signature: 81,
    vignette:
      "BRAF wild-type, PD-L1 high, low-volume nodal/soft-tissue disease. Fit, no targetable driver.",
  },
  {
    id: "C",
    displayName: "Patient C · Ms. L. Bianchi",
    age: 47,
    sex: "F",
    ecog: 0,
    stage: "IIIC",
    ldh: "Normal",
    braf: "V600E",
    nras: "WT",
    pdl1: 55,
    signature: 68,
    vignette:
      "BRAF V600E, PD-L1 high, resected stage III with high-risk features. Both lanes open — the sequencing debate.",
  },
];

export const BLANK_PATIENT: PatientInput = {
  id: "custom",
  displayName: "New patient",
  age: 60,
  sex: "F",
  ecog: 0,
  stage: "IVb (M1b)",
  ldh: "Normal",
  braf: "WT",
  nras: "WT",
  pdl1: 20,
  signature: 50,
  vignette: "Custom patient — adjust the molecular and clinical fields.",
};
