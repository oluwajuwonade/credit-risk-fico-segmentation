<!-- PORTFOLIO-CONTEXT
Oluwajuwon Adediji | Data & Quantitative Analyst | Quantitative Risk Analytics
Portfolio: https://oluwajuwonade.vercel.app
-->

# Credit Risk Analytics & FICO Segmentation

> **Decision problem:** How can borrower-level risk be modelled and converted into interpretable credit-risk segments?

A reproducible quantitative-risk project demonstrating probability-of-default modelling and dynamic FICO score bucketing.

## Executive summary

The project implements two core retail-credit-risk techniques:

1. **Probability of default (PD) modelling** using logistic regression.
2. **FICO score segmentation** using dynamic programming to identify statistically distinct rating buckets.

The repository reports results on a fully synthetic, seeded borrower dataset and includes implementation code, an executed notebook, outputs, and tests.

## Results on the synthetic dataset

| Metric | Result |
|---|---:|
| Portfolio size | 50,000 synthetic borrowers |
| Overall default rate | 13.3% |
| Test AUC | 0.70 |
| FICO rating buckets | 8 |
| Default-rate monotonicity | Strictly decreasing |

These values describe the synthetic data-generating process and are **not claims about any real credit portfolio**.

## Methodology

`Synthetic borrower data → Feature inspection → Train/test split → Logistic PD model → Model evaluation → Dynamic FICO bucketing → Rating validation → Risk interpretation`

### PD modelling

A logistic-regression model estimates borrower-level probability of default from FICO score, debt-to-income, credit lines, and employment tenure.

### FICO bucketing

Instead of naive equal-width or equal-frequency bins, candidate FICO boundaries are optimized using dynamic programming. Each candidate bucket is treated as a binomial segment and evaluated through log-likelihood, subject to a minimum bucket size.

## Validation

The repository includes:

- Model performance evaluation
- Monotonicity verification
- Reproducible synthetic data generation
- Unit tests for the bucketing algorithm
- Explicit methodological limitations

## Repository structure

```text
credit-risk-fico-segmentation/
├── data/
├── notebooks/
├── src/
├── tests/
├── outputs/
├── README.md
└── requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
jupyter notebook notebooks/credit_risk_segmentation.ipynb
python tests/test_fico_bucketing.py
```

## Important limitations

- The PD model uses a small four-feature specification for demonstration.
- Synthetic-data performance should not be interpreted as production model performance.
- Rating boundaries should be fit on training data and validated out-of-time in a real deployment.
- Real credit-risk work would typically add calibration analysis, model comparison, stability monitoring, and governance controls.

## Portfolio role

**Tier 1 — Flagship Quantitative / Risk Analytics**

This project demonstrates statistical modelling, risk segmentation, algorithmic reasoning, reproducibility, testing, and quantitative communication.

## Related projects

- [Financial Planning & Scenario Modelling](https://github.com/oluwajuwonade/financial-modelling-starter-system)
- [Data Quality & Analytics Assurance](https://github.com/oluwajuwonade/data-quality-audit-toolkit)
- [AI Research & Evaluation Framework](https://github.com/oluwajuwonade/ai-research-evaluation-system)

## Author

**Oluwajuwon Adediji**  
Data & Quantitative Analyst | Quantitative Risk Analytics
