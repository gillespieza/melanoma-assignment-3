## 6. Phase 6: Clinical Utility & Decision Curve Analysis

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Conducting Decision Curve Analysis (DCA), calculating Net Benefit across threshold probabilities ($p_t = 0.1 – 0.9$), and evaluating Number Needed to Treat (NNT).
> - **Why we are doing it**: High AUC-ROC does not guarantee clinical usefulness. DCA evaluates whether using a model to make treatment decisions produces greater net clinical benefit than empirical 'Treat All' or 'Treat None' strategies.
> - **What question it answers**: Does deploying the Q5 stratification model in clinical practice yield superior Net Benefit and reduce unnecessary treatment toxicities?

Phase 6 quantifies real-world clinical utility across $N = 326$ patients using the Net Benefit formula:

$$\text{Net Benefit} = \frac{\text{True Positives}}{N} - \left( \frac{\text{False Positives}}{N} \right) \times \left( \frac{p_t}{1 - p_t} \right)$$

Across all clinically relevant threshold probabilities ($p_t = 0.2 – 0.6$), the Q5 decision system achieves higher Net Benefit than treating all patients empirically or relying on single-gene `CD274` (PD-L1) cutoffs. Additionally, the model significantly lowers the Number Needed to Treat (NNT) to achieve one objective response.

### Key Takeaways
- **Superior Net Benefit**: Guided treatment decisions add positive net clinical benefit across all realistic threshold ranges.
- **Toxicity Reduction**: Prevents predicted non-responders from undergoing ineffective immunotherapy monotherapy.
