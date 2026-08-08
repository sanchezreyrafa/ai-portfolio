# Cost Analysis

Rough compute/memory/storage cost profile for the fraud detection endpoint, based on measurements taken directly from this repo . See `README.md`'s Infrastructure Notes for the raw dataset load numbers this builds on.

## Training

- **Dataset footprint**: the merged transaction+identity dataset is 590,540 rows × 437 columns, ~2.5GB in memory once loaded as a pandas DataFrame — but only 96MB on disk as parquet (~26x smaller), since parquet's columnar compression handles the dataset's heavy nulls and repeated categorical values well. This gap matters for cost planning: storage is cheap, but any training or batch-scoring job needs to budget for the in-memory footprint, not the on-disk size.
- **Memory**: 8GB minimum, 16GB recommended for the full training notebook (per README) — driven by holding the merged frame plus the one-hot-expanded feature matrix simultaneously during preprocessing.
- **Compute**: training is CPU-only and fast — the full grid search in `03_training.ipynb` (12 XGBoost fits: 2 baseline + 8-value `scale_pos_weight` sweep + 1 refit of the winning `spw=5` config for threshold analysis + 1 final combined-data retrain) completes in well under 10 minutes on a single laptop CPU core. No GPU is required at this data scale; this is not a training-cost-dominated workload. GPU acceleration would start to pay off in two scenarios: if the dataset grew roughly 10x (tree-building time scales with rows × features, and GPU-backed `tree_method='hist'` cuts that meaningfully at larger volumes), or if the model itself moved from gradient-boosted trees to a neural network, where GPU parallelism is the dominant cost lever rather than an optional speedup.
- **Note on retraining cost**: the 12 fits above are the one-time (or periodic, if re-tuning) cost of the hyperparameter search itself — finding the right `scale_pos_weight` and threshold. A routine production retrain (e.g. the sliding-window retraining the notebook recommends to keep up with fraud-pattern drift) only needs **1 fit**, using the already-known `scale_pos_weight=5` config on the fresh data window — not the full sweep.

## Inference

- **Model size**: `model.json` is 606KB — trivial to load, cache in memory, or ship in a container image.
- **Measured latency**: 200 sequential requests through `run_inference()` (in-process, no network) averaged **~49ms/request**.
- **Where the time goes**: this is preprocessing-bound, not model-bound. XGBoost's own `predict_proba` call on a single row with 216 features is sub-millisecond territory; the cost is in `_preprocess()`'s repeated single-column `df[col] = ...` assignments on a one-row DataFrame (pandas flags this directly as a `PerformanceWarning: DataFrame is highly fragmented`, since each assignment currently triggers a fresh internal reallocation). This is a known, fixable bottleneck — batching the derived/encoded columns into a single `pd.concat` instead of ~30 sequential inserts would cut this substantially — not a case that needs more hardware to fix.
- **Throughput implication**: at ~49ms/request, a single API worker serially handles roughly 20 requests/second. For a fraud-detection endpoint sitting in a real-time authorization path, that's adequate for a low-to-moderate traffic service but would need either the preprocessing fix above or horizontal scaling (multiple uvicorn workers) before handling meaningful production volume.

## Hosting (rough order of magnitude)

The service is lightweight enough that infrastructure cost is dominated by choice of hosting model, not raw compute:
- **Small always-on VM** (e.g. 1-2 vCPU, 2-4GB RAM — comfortably covers the 606KB model plus FastAPI/uvicorn/pandas/xgboost overhead): rough low tens of USD/month on any major cloud provider.
- **Serverless/container-per-request** (e.g. Cloud Run, Lambda with a container image): likely cheaper for low/bursty traffic given the sub-second execution time per request, at the cost of cold-start latency on the first request after idle.

These are intentionally order-of-magnitude estimates, not live pricing quotes — the point is that this model is small and cheap to serve; the real cost driver in this pipeline is the training-time data volume (2.5GB in-memory, 16GB RAM recommended), not inference hosting.