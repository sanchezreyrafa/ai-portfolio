# EDA Disputes for Credit cards

This project is an EDA notebook on the CFPB US Consumer Finance Complaints dataset — filtered for credit card complaints

## What is analysed in this EDA?
1. How imbalanced is the dispute outcome label? — defines training strategy and threshold calibration
2. Which complaint types and sub-products carry the most predictive signal for dispute outcome?
3. Are there temporal patterns in complaint volume or resolution rates worth engineering as features?
4. What is the missing data profile across key columns? — informs preprocessing pipeline design
5. What is the cardinality of categorical features (company, product, issue)? — affects encoding strategy and memory footprint
6. How does response time (date received → date sent) correlate with dispute outcome? — latency as a domain feature
7. What are the memory and I/O characteristics of loading and processing this dataset? — benchmark DataFrame size, dtypes footprint, and read time as a baseline for pipeline design

## Dataset
Source: US Consumer Finance Complaints — CFPB via Kaggle
Download: https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints

**Original Dataset size:** 560.8 MB → 555,957 rows × 18 columns
**Final Dataset size:** 22.9 MB → 64,388 rows × 9 columns
**CSV read time:** ~1.5s

## How to run it
1. Download the dataset from Kaggle and place it at `data/consumer_complaints.csv`
2. Download `EDA_Disputes.ipynb`
3. Go to https://jupyter.org/try-jupyter/lab/
4. Import both files
5. Run the cells

## Infrastructure Notes

**Memory footprint:**
- Raw load: 560.8 MB (555,957 rows × 18 columns, all `object` dtype strings)
- After filtering to credit card complaints and dropping zero-signal/leakage columns: 22.9 MB (64,388 rows × 9 columns) — an ~24x reduction
- Row reduction driven by product filter (credit card only) and data cleaning (null removal, outlier removal on `response_days`)
- Column reduction from dropping fields with >80% nulls not captured for credit card complaints, an identifier with no signal (`complaint_id`), and two columns with data leakage (`company_response_to_consumer`, `timely_response`) — generated after the company processes the complaint, so unavailable at prediction time

**I/O:**
- CSV read time: ~1.5s — acceptable for a 560 MB file at this scale

**Dtype observation:**
- All string columns are stored as `object` dtype in pandas, which uses Python object pointers (expensive at scale)
- Converting to `pd.Categorical` stores values as integer codes internally and would reduce memory further — a potential optimization not applied in this module