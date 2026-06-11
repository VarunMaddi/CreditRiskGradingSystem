# Credit Risk Grading System

An end-to-end credit risk grading pipeline built on public data sources. Ingests SEC EDGAR financial filings, scores risk factors using FinBERT sentiment analysis, extracts structured credit signals from MD&A sections using an LLM with reasoning, models Probability of Default and Loss Given Default, and produces portfolio-level stress testing, SHAP explainability, and an 8-panel dashboard.

---

## Dashboard

![Dashboard](assets/dashboard.png)

---

## Pipeline

```
UCLA LoPucki BRD              S&P 500 Constituents
(100 Chapter 11 companies)    (100 healthy companies)
         │                            │
         └──── Year-Matched Universe ─┘
                      │
                      ▼
           SEC EDGAR XBRL API
           (11 raw financial metrics)
                      │
                      ▼
     13 Financial Ratios + Altman Z-Score + Ohlson O-Score
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   SEC EDGAR 10-K Filing      SEC EDGAR 10-K Filing
   Item 1A — Risk Factors     Item 7 — MD&A
   (1500 words extracted)     (5000 words, 4500 tokens)
          │                        │
          ▼                        ▼
   FinBERT Sentiment          OpenRouter gpt-oss-120b
   (ProsusAI/finbert)         (reasoning enabled)
   1 continuous score         6 numeric credit scores
          │                        │
          └───────────┬────────────┘
                      ▼
           Master Dataset
           166 companies × 25 features
                      │
            ┌─────────┴──────────┐
            ▼                    ▼
   Calibrated LightGBM       Logistic Regression
   + CalibratedClassifierCV  (baseline)
   CV AUC: 0.983
            │
            ▼
   GBR LGD Model
            │
            ▼
   EL = PD × LGD × EAD
            │
     ┌──────┴───────┐
     ▼              ▼
Stress Testing   SHAP Values
4 macro scenarios per-company
     │
     ▼
8-Panel Matplotlib Dashboard
```

---

## Results

| Metric | Value |
|---|---|
| Companies | 166 (77 defaulted, 89 healthy) |
| Features | 25 |
| Default rate | 46.4% |
| CV AUC (5-fold) | 0.983 ± 0.011 |
| CV Average Precision | 0.979 |
| CV Brier Score | 0.060 |
| CV Log Loss | 0.195 |
| Time-based holdout AUC | 0.955 |
| Recession EL delta | +11.0% |
| Severe stress EL delta | +15.6% |
| FinBERT SHAP contribution | 8.4% |
| LLM SHAP contribution | 13.0% |

---

## Features (25 total)

### Financial Ratios — 13
Engineered from SEC EDGAR XBRL data, clipped to remove outliers:

| Feature | Formula |
|---|---|
| leverage_ratio | Liabilities / Assets |
| debt_to_equity | LongTermDebt / StockholdersEquity |
| debt_to_assets | LongTermDebt / Assets |
| current_ratio | AssetsCurrent / LiabilitiesCurrent |
| working_capital | AssetsCurrent − LiabilitiesCurrent |
| roa | NetIncome / Assets |
| net_profit_margin | NetIncome / Revenue |
| asset_turnover | Revenue / Assets |
| interest_coverage | OperatingIncome / InterestExpense |
| ocf_to_debt | OperatingCashFlow / LongTermDebt |
| ocf_to_assets | OperatingCashFlow / Assets |
| ocf_margin | OperatingCashFlow / Revenue |
| log_assets | log(1 + Assets) |

Missing Liabilities imputed as Assets − StockholdersEquity where available.

### Academic Credit Scores — 5

**Altman Z-Score (1968):** `Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5`
- Zones: Safe (Z > 2.99), Grey (1.81–2.99), Distress (Z < 1.81)
- Defaulted mean Z: 0.58 vs healthy mean Z: 1.55 ✓

**Ohlson O-Score (1980):** Logistic default probability from 9 balance sheet inputs

Binary flags: `z_distress`, `z_safe`

### FinBERT Sentiment — 1

FinBERT (ProsusAI/finbert) scores Item 1A (Risk Factors) only. MD&A excluded because management tone is inherently optimistic even in distress — Item 1A uses legally required negative language and provides cleaner signal.

| Feature | Description |
|---|---|
| finbert_1a_net_negative | 1A negative score − 1A positive score |

Truncated to 250 words per FinBERT's 512 token limit. SHAP contribution: **8.4%**.

### LLM Numeric Scores — 6

OpenRouter `gpt-oss-120b` with reasoning enabled scores the full MD&A section. tiktoken used for precise 4500-token truncation (Groq free tier limit). Sequential inference with 2s sleep.

| Score | Range | Description |
|---|---|---|
| going_concern_severity | 0–10 | 2=boilerplate, 6=substantial doubt, 10=imminent filing |
| debt_risk_score | 0–10 | 2=manageable, 6=near covenant limit, 10=default |
| liquidity_risk_score | 0–10 | 2=strong cash, 6=reliant on credit lines, 10=unable to pay |
| management_distress | 0–10 | 2=confident/strong guidance, 8=cautious/poor outlook |
| restructuring_risk | 0–10 | 2=no mention, 6=underway, 10=bankruptcy filing |
| revenue_distress | 0–10 | 2=strong growth, 6=flat/slight decline, 10=severe decline |

SHAP contribution: **13.0%**. Full MD&A context (4500 tokens vs 400 words) was critical — compressed inputs produced near-zero model contribution.

---

## Models

### PD Model

**Logistic Regression** (baseline):
- Median imputation → StandardScaler → LogisticRegression
- class_weight="balanced", L2 C=0.1
- Holdout AUC: 0.941

**Calibrated LightGBM** (final):
- Regularized: max_depth=3, num_leaves=7, min_child_samples=15
- L1 reg_alpha=1.0, L2 reg_lambda=2.0
- Wrapped in CalibratedClassifierCV (isotonic regression, cv=5)
- scale_pos_weight for class imbalance
- CV AUC: 0.983 ± 0.011

**Why calibration matters:** LightGBM probabilities are often poorly ranked — calibration maps raw scores to true empirical probabilities. Required for EL = PD × LGD × EAD to be meaningful.

**Time-based validation:** Train ≤2017, test ≥2018 — prevents future patterns leaking into training, the regulatory standard for credit model validation.

### LGD Model

Gradient Boosting Regressor trained on defaulted companies only. LGD proxy constructed from balance sheet structure — base rate 60% adjusted for leverage (+), liquidity (−), and profitability (−). Uses LLM `debt_risk_score` and `liquidity_risk_score` as additional features. Clipped to [0.10, 0.95].

### Expected Loss

```
EL_rate = PD × LGD
EL      = PD × LGD × EAD
```

EAD proxied as total assets from EDGAR XBRL.

---

## SHAP Explainability

| Rank | Feature | Mean \|SHAP\| | Group |
|---|---|---|---|
| 1 | working_capital | 0.704 | Ratio |
| 2 | roa | 0.585 | Ratio |
| 3 | log_assets | 0.566 | Ratio |
| 4 | ocf_to_assets | 0.454 | Ratio |
| 5 | going_concern_severity | 0.433 | LLM |
| 6 | finbert_1a_net_negative | 0.324 | FinBERT |
| 7 | debt_to_assets | 0.322 | Ratio |
| 8 | ocf_to_debt | 0.297 | Ratio |
| 9 | ohlson_o | 0.056 | Z-Score |
| 10 | debt_risk_score | 0.054 | LLM |

**Feature group contributions:**
- Ratio: 76.6%
- LLM: 13.0%
- FinBERT: 8.4%
- Z-Score: 2.0%

**J.C. Penney (PD=1.000):**
```
roa                    +0.803  ↑ increases PD
debt_to_assets         +0.624  ↑ increases PD
ocf_to_assets          +0.620  ↑ increases PD
log_assets             +0.529  ↑ increases PD
```

**Fortive (PD=0.000):**
```
roa                    -0.571  ↓ decreases PD
going_concern_severity -0.427  ↓ decreases PD
ocf_to_assets          -0.370  ↓ decreases PD
finbert_1a_net_negative -0.357 ↓ decreases PD
```

---

## Stress Testing

Financial ratio shocks combined with NLP sentiment shocks — recession scenario degrades both financial and LLM/FinBERT features simultaneously:

| Scenario | Shocks Applied |
|---|---|
| Rate Shock (+200bps) | interest_coverage −1.5, leverage +0.10, debt_risk +1.5, liquidity_risk +1.0 |
| Recession (GDP −3%) | roa −0.03, net_profit_margin −0.05, finbert_1a_net_negative +0.10, management_distress +1.5, revenue_distress +2.0 |
| Severe Stress | Both combined |

| Scenario | Avg PD | EL Rate | ΔEL |
|---|---|---|---|
| Base Case | 0.462 | 0.305 | — |
| Rate Shock (+200bps) | 0.471 | 0.311 | +1.8% |
| Recession (GDP −3%) | 0.517 | 0.339 | +11.0% |
| Severe Stress (Both) | 0.540 | 0.353 | +15.6% |

---

## Dataset Construction

**Bias elimination:** Healthy S&P 500 companies assigned the same year distribution as defaulted companies — eliminating size and time-period bias that inflates model performance.

**Target year:** Defaulted companies use financials from 2 years before Chapter 11 filing. Healthy companies assigned matching years.

**LLM text scope:** Full Item 7 MD&A (up to 5000 words extracted, 4500 tokens sent to LLM). Earlier versions using 400-word truncation produced near-zero LLM contribution — full context was critical.

**Known limitation:** Dataset is small (166 companies after cleaning). PD scores remain somewhat bimodal — companies cluster near 0 or 1 with sparse intermediate values, reflecting genuine class separation rather than pure overfitting. Regularization (L1/L2, constrained tree depth) and isotonic calibration applied to improve intermediate score quality.

---

## Notebook Structure

| Cell | Purpose |
|---|---|
| 1 | Setup — installs, imports, Drive mount, tiktoken encoder |
| 2 | Universe — LoPucki filter + S&P 500, year-matched 200 companies |
| 3 | EDGAR pull — XBRL API, 11 raw metrics, Drive cache |
| 4 | Ratio engineering — 13 ratios + Altman Z + Ohlson O, outlier clipping |
| 5 | 10-K extraction — Item 1A (1500w) + Item 7 (5000w), 2nd-match regex skips TOC |
| 6 | FinBERT — Item 1A scored at 250-word chunks, cached to Drive |
| 7 | OpenRouter LLM — full MD&A (4500 tokens), gpt-oss-120b with reasoning |
| 8 | Merge — left join ratio + FinBERT + LLM, median imputation for missing |
| 9 | PD + LGD + EL modeling — time-based split, calibrated LightGBM, 5-fold CV |
| 10 | SHAP — TreeExplainer on base estimator, group contributions, company explanations |
| 11 | Stress test + industry concentration — financial + NLP shocks |
| 12 | Dashboard — 8-panel matplotlib, saved as PNG |

---

## Data Sources

| Source | Purpose | Cost |
|---|---|---|
| [SEC EDGAR XBRL API](https://data.sec.gov/api/xbrl/) | Financial statements | Free |
| [SEC EDGAR Submissions API](https://data.sec.gov/submissions/) | 10-K filing URLs | Free |
| [UCLA LoPucki BRD](https://lopucki.law.ucla.edu) | Chapter 11 bankruptcy labels | Free |
| [S&P 500 Constituents](https://github.com/datasets/s-and-p-500-companies) | Healthy company universe | Free |
| [OpenRouter](https://openrouter.ai) | gpt-oss-120b inference | Free tier |
| [HuggingFace](https://huggingface.co/ProsusAI/finbert) | FinBERT model | Free |

---

## Setup

```bash
pip install lightgbm groq transformers torch shap scikit-learn \
            plotly matplotlib tiktoken requests openpyxl joblib \
            beautifulsoup4 sec-edgar-downloader
```

### API Keys Required

| Key | Source | Usage |
|---|---|---|
| OpenRouter API key | [openrouter.ai](https://openrouter.ai) — free | Cell 7 — LLM scoring |
| SEC EDGAR | No key — add email to User-Agent header | Cells 3, 5 |

### Running

1. Open `CreditRiskGradingSystem.ipynb` in Google Colab
2. Mount Google Drive — all outputs persist across sessions
3. Add OpenRouter API key in Cell 7 (`OPENROUTER_KEY = "..."`)
4. Add your email in Cell 1 (`HEADERS = {'User-Agent': 'name email@domain.com'}`)
5. Download LoPucki BRD from [lopucki.law.ucla.edu](https://lopucki.law.ucla.edu) — upload when prompted by Cell 2
6. Run all cells top to bottom

**Estimated runtime:**
- First run: ~50 minutes (EDGAR pull ~5min, 10-K fetch ~15min, LLM scoring ~25min)
- Subsequent runs: ~5 minutes (all API responses cached to Drive)

---

## Project Structure

```
CreditRiskGradingSystem/
│
├── CreditRiskGradingSystem.ipynb
├── README.md
├── assets/
│   └── dashboard.png
│
└── data/                              # Generated on run — not committed
    ├── universe_200.csv
    ├── edgar_raw.csv
    ├── ratio_dataset.csv
    ├── finbert_signals.csv
    ├── llm_signals.csv
    ├── master_dataset.csv
    ├── shap_values.csv
    ├── shap_importance.csv
    ├── stress_results.csv
    ├── industry_concentration.csv
    ├── feature_config.json
    ├── calibration_curve.png
    ├── dashboard.png
    ├── lgb_pd_model.pkl
    ├── lgd_model.pkl
    └── lr_pd_model.pkl
```

### .gitignore

```
data/
cache/
*.pkl
__pycache__/
.ipynb_checkpoints/
```

---

## Methodology Notes

**Why Item 1A for FinBERT, Item 7 for LLM:** FinBERT was trained on financial news and analyst reports — it reads tone, not financial reality. MD&A uses professionally optimistic language even in distress (e.g. Quorum Health reports $134M pre-tax loss but management writes about "growth opportunities"). Item 1A uses legally required negative language — FinBERT reads this correctly. The LLM with reasoning can parse actual financial numbers from MD&A (losses, covenant status, cash burn) and score them accurately.

**Why full MD&A context matters:** Initial tests with 400-word MD&A truncation produced near-zero LLM model contribution (SHAP ≈ 0%). Expanding to 4500 tokens (tiktoken-precise) covering the full financial discussion increased LLM SHAP contribution to 13.0% — the same model, the same prompt, entirely different signal quality.

**Calibration:** LightGBM predict_proba outputs are often poorly calibrated — correct ranking but wrong probabilities. CalibratedClassifierCV with isotonic regression fits a monotonic mapping from raw scores to true empirical probabilities. Essential for EL = PD × LGD × EAD to produce meaningful dollar amounts.
