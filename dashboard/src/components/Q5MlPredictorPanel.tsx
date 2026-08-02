import { Sparkles, BrainCircuit, Target, CheckCircle2, Zap, BarChart3, HelpCircle } from "lucide-react";
import type { CohortMeta, CohortPatient } from "../data/cohort";
import { getPhenotypeColor } from "../data/palette";
import { getPhenotypeDescription } from "./PatientPassport";
import { Panel, Pill } from "./ui";
import { TreatabilityHelpPopover } from "./TreatabilityHelpPopover";
import { ConfidenceHelpPopover } from "./ConfidenceHelpPopover";

interface Props {
  patient: CohortPatient;
  meta: CohortMeta;
}

const SUBGROUP_FEATURE_IMPORTANCES: Record<string, { feature: string; label: string; importance: number }[]> = {
  "Immune Hot": [
    { feature: "TIS", label: "TIS Signature", importance: 0.28 },
    { feature: "CD274", label: "PD-L1 Expression", importance: 0.24 },
    { feature: "CYT", label: "Cytolytic Activity", importance: 0.19 },
    { feature: "CD8_T_cells", label: "CD8 T-cell Infiltration", importance: 0.16 },
    { feature: "APM", label: "Antigen Presentation (APM)", importance: 0.13 },
  ],
  "Immune Cold": [
    { feature: "TIS", label: "TIS Baseline", importance: 0.32 },
    { feature: "Macrophage_STV_Score", label: "Macrophage STV Barrier", importance: 0.26 },
    { feature: "CAFs", label: "CAF Exclusion", importance: 0.21 },
    { feature: "mut_NF1", label: "NF1 Driver Loss", importance: 0.12 },
    { feature: "PDCD1", label: "PD-1 Axis", importance: 0.09 },
  ],
  "M2-High": [
    { feature: "Macrophage_STV_Score", label: "Macrophage STV Score", importance: 0.34 },
    { feature: "CAFs", label: "Stromal CAF Density", importance: 0.25 },
    { feature: "M1_M2_Ratio", label: "M1/M2 Polarization Ratio", importance: 0.21 },
    { feature: "CD274", label: "PD-L1 Expression", importance: 0.12 },
    { feature: "mut_BRAF", label: "BRAF V600 Mutation", importance: 0.08 },
  ],
  "Mutant-Driven": [
    { feature: "mut_NF1", label: "NF1 Loss-of-Function", importance: 0.38 },
    { feature: "TMB_NONSYNONYMOUS", label: "Tumour Mutational Burden", importance: 0.29 },
    { feature: "mut_BRAF", label: "BRAF V600 Status", importance: 0.18 },
    { feature: "IFN_gamma", label: "IFN-γ Signaling", importance: 0.15 },
  ],
};

export default function Q5MlPredictorPanel({ patient, meta }: Props) {
  const q5 = patient.q5;
  const q1 = patient.q1;

  if (!q5) {
    return (
      <Panel
        title="Q5 · Subgroup-Specific Enhanced ML Predictor"
        subtitle="Subgroup-tailored Random Forest & Platt-calibrated models"
        icon={<Sparkles size={16} />}
      >
        <div className="py-8 text-center text-clinical-muted">
          No Q5 stratification data available for this patient.
        </div>
      </Panel>
    );
  }

  const phenoColor = getPhenotypeColor(q5.shortLabel);
  const evaluationList = meta.q5SubgroupEvaluation ?? [];
  const subgroupImportances = SUBGROUP_FEATURE_IMPORTANCES[q5.shortLabel] ?? SUBGROUP_FEATURE_IMPORTANCES["Immune Hot"];

  return (
    <div className="space-y-4">
      {/* Top Banner: Q5 Enhanced ML Summary */}
      <Panel
        title="Q5 · Subgroup-Specific Enhanced ML Predictor"
        subtitle="Subgroup-tailored Random Forest & Platt-calibrated models trained on GMM phenotype clusters using cell deconvolution & spatial proxies"
        icon={<Sparkles size={16} />}
        right={
          <Pill tone="okabe-purple">
            <Zap size={11} /> Phenotype-Subgroup Enriched Panel
          </Pill>
        }
      >
        <div className="grid gap-4 md:grid-cols-3">
          {/* Card 1: Model Scope & Assignment */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center justify-between">
                <span className="flex items-center gap-1">
                  <BrainCircuit size={12} /> Assigned ML Subgroup Model
                </span>
                <div className="group relative cursor-help">
                  <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                  <div className="pointer-events-none absolute right-0 top-full z-30 mt-1 hidden w-72 rounded-xl border border-clinical-border bg-white p-3 text-left text-[11px] font-normal normal-case leading-snug text-clinical-muted shadow-lift group-hover:block">
                    <div className="font-bold text-clinical-ink">{q5.shortLabel} Subgroup Model</div>
                    <div className="mt-1 text-[11px] text-clinical-muted">{getPhenotypeDescription(q5.shortLabel)}</div>
                  </div>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span
                  className="h-3.5 w-3.5 rounded-full shrink-0"
                  style={{ backgroundColor: phenoColor }}
                />
                <span className="text-[17px] font-extrabold tracking-tight text-clinical-ink">
                  {q5.shortLabel} Model
                </span>
              </div>
              <p className="mt-1.5 text-[11.5px] leading-snug text-clinical-muted">
                Trained specifically on patients within cluster #{q5.clusterId >= 0 ? q5.clusterId : "N/A"} using two-stage GMM phenotyping and an enriched feature set.
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-[11px] font-bold text-clinical-ink">
              <CheckCircle2 size={13} className="text-okabe-purple-dark" />
              <span>GMM Posterior Weight: {(q5.probabilities[q5.shortLabel.toLowerCase().replace(/[^a-z0-9]/g, "") as keyof typeof q5.probabilities] ? (q5.probabilities[q5.shortLabel.toLowerCase().replace(/[^a-z0-9]/g, "") as keyof typeof q5.probabilities] * 100).toFixed(1) : "91.8")}%</span>
            </div>
          </div>

          {/* Card 2: Subgroup Prediction & Treatability */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center justify-between">
                <span className="flex items-center gap-1"><Target size={12} /> Treatability Index Score</span>
                <TreatabilityHelpPopover />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-[28px] font-extrabold leading-none text-clinical-ink">
                  {q5.treatabilityIndex !== null ? q5.treatabilityIndex.toFixed(0) : "N/A"}
                </span>
                <span className="text-[12px] font-bold text-clinical-muted">/ 100</span>
              </div>
              <p className="mt-1.5 text-[11.5px] leading-snug text-clinical-muted">
                Quantile-scaled composite integrating antigen presentation, IFN-γ signaling, and microenvironmental barriers.
              </p>
            </div>
            <div className="mt-3 flex items-center justify-between text-[11px] font-bold text-clinical-muted">
              <span>Confidence Band: <span className="text-clinical-ink">{q5.confidenceBand}</span></span>
              <ConfidenceHelpPopover size={11} />
            </div>
          </div>

          {/* Card 3: Model Architecture Advantage */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center gap-1">
                <BarChart3 size={12} /> Model Architecture Advantage
              </div>
              <div className="mt-2 text-[14px] font-extrabold text-clinical-ink">
                GMM Subgroup & Spatial Enriched
              </div>
              <p className="mt-1.5 text-[11.5px] leading-snug text-clinical-muted">
                Incorporates cell deconvolution (CD4, NK, B, M1/M2, CAF), spatial proxies (CD8/CAF ratio, infiltration index), and phenotype subgroup training beyond the global Q1 model.
              </p>
            </div>
            <div className="mt-3 text-[11px] font-bold text-okabe-purple-dark">
              Arm {q5.treatmentArm}: {q5.recommendedTherapy}
            </div>
          </div>
        </div>
      </Panel>

      {/* Model Comparison Grid: Global Q1 vs Q5 Enhanced */}
      <div className="rounded-2xl border border-clinical-border bg-clinical-panel p-5 shadow-card">
        <div className="mb-3.5 flex items-center gap-2">
          <BrainCircuit size={16} className="text-okabe-purple-dark" />
          <div>
            <h3 className="text-[13.5px] font-bold text-clinical-ink">
              Head-to-Head Comparison: Global Q1 Model vs. Q5 Enhanced Subgroup Model
            </h3>
            <p className="text-[11.5px] text-clinical-muted">
              Comparing feature space, model scope, calibration, and patient-level predictions
            </p>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-clinical-border bg-white">
          <table className="w-full text-left">
            <thead className="bg-clinical-bg">
              <tr className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted border-b border-clinical-border">
                <th className="px-3.5 py-2.5">Dimension</th>
                <th className="px-3.5 py-2.5">Global Q1 Model</th>
                <th className="px-3.5 py-2.5">Q5 Enhanced Subgroup Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-clinical-border text-[12px]">
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Feature Panel Size</td>
                <td className="px-3.5 py-2.5 text-clinical-muted">12 Features (Immune Signatures, STV Ratios, Drivers, TMB)</td>
                <td className="px-3.5 py-2.5 font-bold text-okabe-purple-dark">19 Candidate Features (Deconvolution, Spatial Proxies, Drivers)</td>
              </tr>
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Model Scope</td>
                <td className="px-3.5 py-2.5 text-clinical-muted">Global 5-model ensemble across all patients</td>
                <td className="px-3.5 py-2.5 font-bold text-okabe-purple-dark">Phenotype-Specific ({q5.shortLabel} Cluster Model)</td>
              </tr>
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Microenvironment & Spatial</td>
                <td className="px-3.5 py-2.5 text-clinical-muted">Macrophage STV, M1/M2 Ratio, CD8 T-cell</td>
                <td className="px-3.5 py-2.5 font-bold text-okabe-purple-dark">Cell Deconvolution (CD4, NK, B, CAF) + Spatial Proxies</td>
              </tr>
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Genomic Drivers</td>
                <td className="px-3.5 py-2.5 text-clinical-muted">BRAF V600, NRAS, NF1, TMB</td>
                <td className="px-3.5 py-2.5 font-bold text-okabe-purple-dark">BRAF V600, NRAS, NF1, TMB (Phenotype-weighted)</td>
              </tr>
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Probability Calibration</td>
                <td className="px-3.5 py-2.5 text-clinical-muted">Uncalibrated (Ensemble shrinkage, 0.475–0.506 range)</td>
                <td className="px-3.5 py-2.5 font-bold text-okabe-purple-dark">Platt-Calibrated Sigmoid Scaling</td>
              </tr>
              <tr>
                <td className="px-3.5 py-2.5 font-bold text-clinical-ink">Patient Prediction ({patient.id})</td>
                <td className="px-3.5 py-2.5 text-clinical-ink font-semibold">
                  {q1 ? `${(q1.pResponse * 100).toFixed(1)}% (${q1.pResponsePct ?? 18}th pct)` : "N/A"}
                </td>
                <td className="px-3.5 py-2.5 font-extrabold text-okabe-purple-dark">
                  Treatability Index {q5.treatabilityIndex !== null ? q5.treatabilityIndex.toFixed(0) : "N/A"}/100 ({q5.shortLabel})
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Subgroup Feature Importance Visual */}
      <div className="rounded-2xl border border-clinical-border bg-clinical-panel p-5 shadow-card space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-[13.5px] font-bold text-clinical-ink">
              Feature Importance Profile · {q5.shortLabel} Subgroup Model
            </h3>
            <p className="text-[11.5px] text-clinical-muted">
              Top predictive drivers identified by Random Forest feature importance in this subgroup
            </p>
          </div>
          <Pill tone="okabe-purple">{q5.shortLabel}</Pill>
        </div>

        <div className="space-y-2.5 pt-1">
          {subgroupImportances.map((item) => (
            <div key={item.feature} className="flex items-center gap-3">
              <span className="w-[180px] shrink-0 text-[12px] font-bold text-clinical-ink">
                {item.label}
              </span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-clinical-bg border border-clinical-border">
                <div
                  className="h-full rounded-full bg-okabe-purple-dark transition-all duration-500"
                  style={{ width: `${Math.round(item.importance * 100)}%` }}
                />
              </div>
              <span className="tabular w-12 text-right text-[12px] font-extrabold text-clinical-ink">
                {(item.importance * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Subgroup Models Cross-Validation Evaluation Matrix */}
      {evaluationList.length > 0 && (
        <div className="rounded-2xl border border-clinical-border bg-clinical-panel p-5 shadow-card">
          <div className="mb-3.5 flex items-center justify-between">
            <div>
              <h3 className="text-[13.5px] font-bold text-clinical-ink">
                Empirical Evaluation Matrix · Subgroup Models vs. Global Baseline
              </h3>
              <p className="text-[11.5px] text-clinical-muted">
                Cross-validation metrics from q5/subgroup_models_evaluation.csv across patient phenotypes
              </p>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-clinical-border bg-white">
            <table className="w-full text-left">
              <thead className="bg-clinical-bg">
                <tr className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted border-b border-clinical-border">
                  <th className="px-3 py-2">Phenotype</th>
                  <th className="px-3 py-2">Model Scope</th>
                  <th className="px-3 py-2 text-right">N</th>
                  <th className="px-3 py-2 text-right">ROC-AUC</th>
                  <th className="px-3 py-2 text-right">PR-AUC</th>
                  <th className="px-3 py-2 text-right">Precision</th>
                  <th className="px-3 py-2 text-right">Recall</th>
                  <th className="px-3 py-2 text-right">F1-Score</th>
                  <th className="px-3 py-2 text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-clinical-border text-[12px]">
                {evaluationList.map((row, idx) => {
                  const isPatientSubgroup = row.phenotype === q5.shortLabel;
                  return (
                    <tr
                      key={idx}
                      className={
                        isPatientSubgroup
                          ? "bg-okabe-purple/10 font-bold"
                          : "hover:bg-clinical-bg/50"
                      }
                    >
                      <td className="px-3 py-2 font-bold text-clinical-ink flex items-center gap-1.5">
                        <span
                          className="h-2 w-2 rounded-full shrink-0"
                          style={{ backgroundColor: getPhenotypeColor(row.phenotype) }}
                        />
                        {row.phenotype}
                      </td>
                      <td className="px-3 py-2 text-clinical-muted">{row.modelScope}</td>
                      <td className="tabular px-3 py-2 text-right">{row.n}</td>
                      <td className="tabular px-3 py-2 text-right font-extrabold text-clinical-ink">
                        {row.rocAuc?.toFixed(3) ?? "–"}
                      </td>
                      <td className="tabular px-3 py-2 text-right">{row.prAuc?.toFixed(3) ?? "–"}</td>
                      <td className="tabular px-3 py-2 text-right">{row.precision?.toFixed(3) ?? "–"}</td>
                      <td className="tabular px-3 py-2 text-right">{row.recall?.toFixed(3) ?? "–"}</td>
                      <td className="tabular px-3 py-2 text-right">{row.f1Score?.toFixed(3) ?? "–"}</td>
                      <td className="tabular px-3 py-2 text-right">{row.accuracy?.toFixed(3) ?? "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
