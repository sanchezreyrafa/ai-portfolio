# ADR-003: XGBoost as baseline model for fraud detection

## Status
Accepted

## Context
Once the data are preprocessed, the most important decision is to choose the model that best fits our features and our financial context.
The 3 models to study are:
- **Binary classification** with polynomic features in a logistic regression
- **Neural network**
- Decision tree -> **XGBoost**

To take this decision we will consider:
- Feature types: most of the features in our data set are numeric
- Scaling: some of our current features need scaling (i.e. TransactionAmt)
- Nulls: 76% of transactions do not have identity data, so, we have a lot of null values
- Feature importance: are we able to explain how important is a feature in the final decision? In a regulated financial environment, model decisions must be explainable to auditors and compliance teams

We also need to handle the card-related categorical features (card4, card6, ProductCD, etc.). XGBoost offers `enable_categorical=True` as a native option, but after evaluation we ruled it out: it is experimental and, critically, it cannot handle unseen categories at inference time — a category present in production but absent from training would cause the model to fail.

## Decision
We will finally use XGBoost based on:
- Feature types: works fine with numeric values
- Scaling: no scaling needed
- Nulls: native null handling
- Feature importance: we are able to explain the feature importance in the final decision
- This algorithm works also fine with non-linear relations.

**Categorical encoding:** all categorical features will be encoded using `encode_card_features`, which maps every low-frequency or unknown value to an `"other"` catch-all bucket before applying ordinal/label encoding. This makes the pipeline robust to unseen categories at inference time, which `enable_categorical=True` cannot guarantee.

## Alternatives Considered
1) Logistic regression
   - Feature types: works fine with numeric values
   - Scaling: scaling needed
   - Nulls: no null handling
   - Feature importance: easy to explain the feature importance
   - To work with non-linear features we need polynomic features but having 217 features → ~23,000 polynomial pairs at degree 2.

2) Neural networks
   - Feature types: good with numeric values
   - Scaling: scaling needed
   - Nulls: no native null handling
   - Feature importance: difficult to explain the feature importance in the final decision. This is a big issue in a regulated environment.
   - It works perfectly with non-linear relations.
   - High training cost

## Consequences
**Pros:**
- Native null handling. We don't need extra work for D features and identity columns.
- No scaling needed - TransactionAmt and other skewed features can go directly.
- Feature importance - we are able to explain the decisions in a highly regulated financial context.
- Categorical encoding with an `"other"` catch-all makes the pipeline safe for production: unknown categories at inference time are handled gracefully instead of crashing.

**Trade-offs:**
- Higher training cost than logistic regression but lower than NN.
- We need to tune some hyperparameters like learning rate, max_depth, n_estimators, etc.
- More complex to explain than logistic regression.
- Encoding categoricals requires maintaining the known-category vocabulary from training (so `encode_card_features` must be fitted on train and applied consistently to validation/test/production).