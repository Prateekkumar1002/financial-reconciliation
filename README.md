# AI Financial Reconciliation System

## Overview

Financial reconciliation verifies that two independent financial records
describe the same underlying transactions.

This project builds an AI-powered reconciliation engine that
automatically matches transactions between: - Bank Statements - Internal
Check Registers

The system uses a hybrid architecture combining deterministic matching
and machine learning similarity matching.

------------------------------------------------------------------------

## System Architecture

Data Ingestion\
→ Data Preprocessing\
→ Unique Amount Matching\
→ Remove Matched Transactions\
→ Machine Learning Similarity Model\
→ Candidate Retrieval (FAISS)\
→ Global Matching (Hungarian Algorithm)\
→ Human Validation Layer\
→ Evaluation (Precision / Recall / F1)

------------------------------------------------------------------------

## Workflow Design

### 1. Data Ingestion

Load bank statements and check register datasets.

### 2. Preprocessing

-   Text normalization
-   Date parsing
-   Amount rounding
-   Transaction type normalization

### 3. Unique Amount Matching

Transactions where the amount appears exactly once in both datasets are
matched automatically.

### 4. Machine Learning Matching

Remaining unmatched transactions are matched using similarity
features: - Text similarity (sentence embeddings) - Amount similarity -
Date proximity - Transaction type match

### 5. Model Training

A Logistic Regression model learns optimal similarity weights from
positive and negative transaction pairs.

### 6. Candidate Retrieval

FAISS vector search retrieves the most likely matching candidates.

### 7. Global Matching

The Hungarian algorithm ensures one‑to‑one reconciliation between
transactions.

### 8. Human Validation

Low‑confidence matches can be reviewed manually.

### 9. Evaluation

Performance metrics: - Precision - Recall - F1 Score

------------------------------------------------------------------------

## Machine Learning Reconciliation System

Key components: - Sentence Transformers for semantic embeddings -
Feature engineering for structured financial signals - Logistic
Regression similarity scoring - FAISS approximate nearest neighbor
search - Hungarian algorithm for optimal assignment

Example configuration:

TEXT_WEIGHT = 0.40\
AMOUNT_WEIGHT = 0.35\
DATE_WEIGHT = 0.20\
TYPE_WEIGHT = 0.05

MATCH_THRESHOLD = 0.6

------------------------------------------------------------------------

## Governance Framework

### Model Governance

-   version controlled models
-   configurable thresholds
-   reproducible training

### Data Governance

-   schema validation
-   audit logs
-   controlled data ingestion

### Auditability

Each reconciliation produces: (bank_transaction_id,
check_transaction_id, confidence_score)

------------------------------------------------------------------------

## Ownership

### Data Owners

Responsible for: - bank statement ingestion - internal ledger accuracy

### Model Owner

Responsible for: - ML model updates - threshold tuning - monitoring
model performance

### Operations Team

Responsible for: - reconciliation monitoring - manual resolution of
unmatched transactions

------------------------------------------------------------------------

## Operational Controls

### Data Quality Checks

-   duplicate detection
-   schema validation
-   transaction integrity checks

### Threshold Controls

Matches below confidence threshold are flagged for manual review.

### Monitoring

System tracks: - execution time - memory usage - match accuracy

------------------------------------------------------------------------

## Repository Structure

financial-reconciliation/

data/ - bank_statements.csv - check_register.csv

src/ - preprocessing.py - unique_matching.py - feature_engineering.py -
similarity_model.py - faiss_index.py - global_matcher.py -
monthly_pipeline.py - evaluation.py - benchmark.py

config.py\
main.py\
requirements.txt\
Dockerfile

------------------------------------------------------------------------

## Running the System

Install dependencies:

pip install -r requirements.txt

Run pipeline:

python main.py

------------------------------------------------------------------------

## Example Results

Unique Matches Found: 286\
Unmatched Transactions: 22\
ML Matches Found: 14

Precision: 1.00\
Recall: 0.974\
F1 Score: 0.986

------------------------------------------------------------------------

## Future Improvements

-   Online learning from validated matches
-   Anomaly detection
-   Real‑time reconciliation APIs
-   Adaptive confidence thresholds

------------------------------------------------------------------------

## Conclusion

This project demonstrates a hybrid AI reconciliation system combining
deterministic matching, machine learning similarity scoring, and human
validation workflows to automate financial transaction reconciliation at
scale.
