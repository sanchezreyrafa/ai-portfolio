# ADR-001: Join Strategy

## Status
Accepted

## Context
The identity dataset has `144233` rows which is much less than the transaction dataset `590540` (76% of transactions do not have identity data). 
The join key is `TransactionId` and we need to make a decision about how to join both: left join VS. inner join

## Decision
Taking into account the good amount of valid data in the transaction dataset, we don't want to lose this info, so we decide to use a left join
- *Pros:* We keep a big amount of valid data to train our model.
- *Cons:* We will have null values where the identity data doesn't join with transaction data.

## Alternatives Considered
Apply an inner join and the result will be a dataset with only the coincidences. 
- *Pros:* We skip null values in a big amount of rows.
- *Cons:* We would lose a lot of data and finally get a 144233 rows dataset.

## Consequences
1) Rows without identity data are encoded as a binary feature `has_identity` during preprocessing, converting the null pattern into a predictive signal (7.8% vs 2.1% fraud rate).
2) This decision allows us to have a big amount of data with valid information to train our model but on the other hand, identity column will have 76% of null values. This requires the model manage null values natively or the preprocessing manage them. We created `has_identity` column and take into account XGBoost can handle this null values internally.