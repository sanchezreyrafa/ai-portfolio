# ADR-002: Chronological split

## Status
Accepted

## Context
Following good dataset practices, we want to split our dataset into `Train, Validation and Test` (60/20/20). We can split the data randomly or chronologically.

## Decision
We will split the dataset chronologically via the TransactionDT
- *Pros:* No nulls in this feature. We don't train the model on future transactions and evaluate it on past ones.
- *Cons:* We need to split the data manually.

## Alternatives Considered
Apply a random dataset split into 3 different groups.
- *Pros:* We can easily split the data set using `train_test_split` using sklearn.
- *Cons:* A random split would introduce temporal data leakage — the model would train on future transactions and evaluate on past ones, producing artificially inflated metrics that would not hold in production.

## Consequences
1) Split strategy: 60/20/20 chronological split based on `TransactionDT` and taking into account a range of 182 days.
   - Train: first 109 days
   - Validation: next 36 days
   - Test: final 36 days

2) In production, the model will always predict future transactions from the training data and this is exactly what the selected split strategy simulate.