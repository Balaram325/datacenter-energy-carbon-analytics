# Data Center GPU Energy Analytics & Carbon Scenario Modelling

**Data science / machine learning portfolio project** based on GPU DCGM telemetry and scheduler-job records. Preserves the original exploration notebook and adds reproducible job-level energy prediction, a baseline comparison, evaluation metrics, and a transparent *hypothetical* carbon-aware shifting scenario.

> **Scope:** The project does NOT demonstrate operational energy optimisation or actual emission savings. The scheduling scenario estimates emissions *if* some workload energy could be moved to cleaner grid hours, with unchanged GPU energy. There is no grid-time series, deadline/capacity constraint model, or production scheduler. Duration is an input feature, so this model is evaluated as a **retrospective job energy estimator**, not a verified before-submission energy predictor.

## Dataset

User-supplied data files, **not bundled in this lightweight GitHub-ready archive**:

- `data/raw/dcgm.csv` — GPU-level summaries: `id_job`, `Node`, `gpu_id`, `energyconsumed_joules`, `totalexecutiontime_sec`, GPU utilisation and other telemetry.
- `data/raw/scheduler_data.csv` — job scheduler metadata, used in `notebooks/data_exploration_01.ipynb` for matching and GPU-count consistency checks.

Check the source and licence/privacy permissions *before* publishing raw records. The original data contains job/user identifiers. The exported notebook preserves the original code and markdown but clears execution outputs, which could otherwise expose identifiers.

## Setup (Windows / VS Code)

Use an existing Python environment that already has pandas, numpy and scikit-learn if possible. No GPU and no large ML model download is needed.

```powershell
# From the repository root
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.analysis --data data/raw/dcgm.csv --output reports
python -m src.optimise --data data/raw/dcgm.csv --output reports
python -m src.visualize --input reports/holdout_predictions.csv --output reports/holdout_scatter.png
```

Copy your own original `dcgm.csv` and `scheduler_data.csv` into `data/raw/` locally before running; these remain gitignored. If you do not have the raw CSV files, only the synthetic unit tests can run.

## Evaluation design

1. Validate nonnegative energies and positive execution time; reject duplicate `(id_job, Node, gpu_id)` telemetry summaries.
2. Aggregate distinct jobs; sum Joules across GPU rows and convert **1 kWh = 3,600,000 J**.
3. Features: GPU count, duration, SM utilisation, memory utilisation, peak GPU memory and PCIe bandwidth summaries. **No direct power or energy columns are used as model features.**
4. Split *by distinct job* (80% train / 20% test) to avoid same-job leakage.
5. Fit median-only baseline and `HistGradientBoostingRegressor` (log-target) using train-only missing-data imputation.
6. Compare holdout **MAE (kWh), RMSE (kWh), and R²**. Save results to `reports/model_metrics.json` and predictions to `reports/holdout_predictions.csv`.

### Measured local run on the uploaded dataset

- 74,849 distinct job records after aggregation; training: 59,879 jobs; holdout: 14,970 jobs.
- **Gradient boosting:** MAE **0.00539 kWh**, RMSE **0.04271 kWh**, R² **0.8713**.
- **Median baseline:** MAE **0.03572 kWh**, RMSE **0.12391 kWh**, R² **−0.0835**.
- Eight synthetic tests passed. See committed `reports/model_metrics.json` for the exact run output. Holdout predictions were generated locally but are not published.

These metrics are for a *single seeded holdout split*, not cross-validation or an external benchmark. The model uses observed job duration, so it must not be portrayed as a reliable pre-scheduling forecast.

The observed energy is taken directly from the dataset; it is *not* claimed as electricity-meter-calibrated whole-data-centre energy. Negative R² is possible and indicates underperformance relative to a mean predictor; compare explicitly against the reported median baseline too.

## Carbon-aware scenario (not measured savings)

Use example assumptions: 0.65 kgCO₂e/kWh current grid intensity, 0.30 kgCO₂e/kWh cleaner period, 20% of workload energy hypothetically flexible.

```
Baseline kgCO2e = total_kWh × baseline_intensity
Shifted kgCO2e = (total_kWh - flexible_kWh) × baseline_intensity
               + flexible_kWh × cleaner_intensity
```

Scenario assumptions are configurable:

```powershell
python -m src.optimise --baseline-intensity 0.65 --cleaner-intensity 0.30 --flexible-fraction 0.20
```

On the uploaded data the example scenario yields **2,657.35 kWh** recorded GPU energy and **186.01 kgCO₂e / 10.77%** hypothetical emissions avoidance, strictly *under the stated assumptions*. These intensities are **illustrative assumptions**, not sourced local grid measurements. This is a sensitivity calculation, not a deployable optimiser or proof of real carbon reductions. In particular, feasibility constraints, task deadlines, additional idle energy, PUE, energy mix and region-specific emission factors are outside scope.

## Repository structure

```
notebooks/data_exploration_01.ipynb   # preserved original exploration
src/analysis.py                      # job aggregation + ML + holdout metrics
src/optimise.py                      # explicit carbon intensity scenario
src/visualize.py                     # diagnostic plot
src/__init__.py
tests/test_pipeline.py                # synthetic validation and scenario tests
requirements.txt
requirements-dev.txt
data/raw/.gitkeep                      # put CSV files here locally
reports/model_metrics.json          # real local holdout results
reports/carbon_scenario.json        # hypothetical scenario results
reports/holdout_scatter.png         # diagnostic prediction plot
reports/.gitkeep
```

## Portfolio use

This demonstrates exploratory analysis of monitoring/scheduling data, group-level data transformations, ML evaluation with a real holdout set, and transparent carbon scenario modelling. Do not claim production deployment, causal optimisation, specific predictive accuracy, or realised emission reductions without additional evidence.

## Additional measured baseline and feature comparison

Run `python -m src.compare_models --data data/raw/dcgm.csv --output reports` to reproduce a second model comparison. Using the same 80/20 job-disjoint split, a log-target Ridge regressor achieved **R² 0.4213, MAE 0.03233 kWh/job**, while histogram gradient boosting achieved **R² 0.8713, MAE 0.00539 kWh/job** (about **83.3% lower MAE** than Ridge). The median-only baseline MAE was 0.03572 kWh/job.

On a seeded subset of 2,500 held-out jobs, feature permutation by mean MAE increase ranked **duration**, **GPU count**, and **memory utilisation** highest. These associations are not causal feature effects; correlated input features can change rankings. Results: `reports/benchmark_comparison.json`. Completed-job duration means this is retrospective energy estimation rather than a pre-scheduling predictor.
