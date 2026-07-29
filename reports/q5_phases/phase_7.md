## 7. Phase 7: 3-Arm Decision Support & Q4 Target Nominations

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Constructing the master 3-arm decision tree and integrating Q2 drug sensitivity data with Q4 DepMap essentiality targets (`AXL`, `MDM2`, `CSF1R`) and LINCS perturbagens.
> - **Why we are doing it**: Patients who fail Arm A (Immunotherapy) require actionable therapeutic alternatives (Arm B Targeted Therapy or Arm C Combination Regimens).
> - **What question it answers**: How does the decision engine route patients into optimal treatment arms, and what helper targets reverse resistance in non-responders?

Phase 7 operationalises the 3-arm clinical decision tree:
- **Arm A (Immunotherapy Monotherapy)**: Assigned to *Immune Hot* patients with predicted response probability $> 70\%$.
- **Arm B (Targeted Therapy)**: Assigned to `BRAF` V600 mutated patients failing Arm A criteria (*Dabrafenib* + *Trametinib*).
- **Arm C (Chemotherapy / Combination Therapy)**: Assigned to non-responders with low Treatability Index scores (*Dacarbazine*). For *M2 Immunosuppressive* non-responders, Q4 DepMap essentiality analysis nominates `CSF1R` (macrophage depletion), `MDM2` (p53 activation), and `AXL` (kinase inhibition) as primary helper drug targets to restore anti-PD-1 sensitivity.

### Key Takeaways
- **Complete Decision Framework**: Provides clear, actionable routing for 100% of incoming melanoma patients.
- **Mechanistic Target Nomination**: Nominates validated helper targets (`CSF1R`, `MDM2`, `AXL`) to overcome specific resistance mechanisms.
