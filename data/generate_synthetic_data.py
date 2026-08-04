"""
generate_synthetic_data.py

Generates a SYNTHETIC consumer-loan dataset used throughout this project.

IMPORTANT: This data is randomly generated (seeded for reproducibility) to
mirror the *structure* of a typical retail-credit dataset (the kind used in
JPMorgan Chase's "Quantitative Research" virtual simulation on Forage). It is
NOT real JPMorgan Chase data and contains no proprietary or confidential
information. It exists purely to demonstrate the analytical methodology
(PD modelling + FICO-based rating bucketing) on a public, shareable dataset.

Run:
    python generate_synthetic_data.py
Output:
    synthetic_loan_data.csv (written to the same directory)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_BORROWERS = 50_000


def generate_loan_book(n: int = N_BORROWERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n + 1)

    # FICO scores: roughly normal, clipped to the valid 300-850 range
    fico_score = rng.normal(loc=660, scale=55, size=n)
    fico_score = np.clip(fico_score, 300, 850).round().astype(int)

    # Tenure in current employment (years)
    years_employed = np.clip(rng.poisson(lam=5, size=n), 0, 40)

    # Annual income (right-skewed, lognormal)
    income = rng.lognormal(mean=10.9, sigma=0.5, size=n).round(2)

    # Number of open credit lines
    credit_lines_outstanding = rng.poisson(lam=2.3, size=n)

    # Existing loan balance, scaled to income
    loan_amt_outstanding = (income * rng.uniform(0.05, 0.55, size=n)).round(2)

    # Other revolving debt tied to number of credit lines
    other_debt = (credit_lines_outstanding * rng.uniform(400, 4500, size=n)).round(2)
    total_debt_outstanding = (loan_amt_outstanding + other_debt).round(2)

    # --- Default probability: logistic function of the drivers above ---
    fico_component = 1 - (fico_score - 300) / 550          # 0 (best) .. 1 (worst)
    dti = total_debt_outstanding / np.maximum(income, 1e-3)  # debt-to-income

    z = (
        -4.35
        + 4.20 * fico_component
        + 1.10 * dti
        + 0.28 * credit_lines_outstanding
        - 0.05 * years_employed
    )
    p_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, p_default)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "credit_lines_outstanding": credit_lines_outstanding,
            "loan_amt_outstanding": loan_amt_outstanding,
            "total_debt_outstanding": total_debt_outstanding,
            "income": income,
            "years_employed": years_employed,
            "fico_score": fico_score,
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_loan_book()
    out_path = __file__.rsplit("/", 1)[0] + "/synthetic_loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Overall default rate: {df['default'].mean():.2%}")
