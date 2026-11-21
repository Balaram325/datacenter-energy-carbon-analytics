"""Job-level GPU energy modelling, based on DCGM per-GPU telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

JOULES_PER_KWH = 3_600_000
# Deliberately exclude powerusage_* and any other energy-derived fields; targets
# are computed by summing measured GPU energy across a job's GPU records.
FEATURES = ["gpu_count", "duration_sec", "sm_utilization_pct", "memory_utilization_pct", "peak_gpu_memory_gb", "pcie_rx_mb", "pcie_tx_mb"]
DCGM_COLUMNS = ["id_job", "Node", "gpu_id", "energyconsumed_joules", "totalexecutiontime_sec", "avgsmutilization_pct", "avgmemoryutilization_pct", "maxgpumemoryused_bytes", "pcierxbandwidth_megabytes_avg", "pcietxbandwidth_megabytes_avg"]


def load_jobs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mandatory = {"id_job", "Node", "gpu_id", "energyconsumed_joules", "totalexecutiontime_sec"}
    missing = mandatory.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing)}")
    for c in set(DCGM_COLUMNS).difference(df.columns):
        df[c] = np.nan
    df = df[DCGM_COLUMNS].copy()
    for c in DCGM_COLUMNS:
        if c not in ("id_job", "Node", "gpu_id"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["id_job", "Node", "gpu_id", "energyconsumed_joules", "totalexecutiontime_sec"])
    df = df.loc[(df.energyconsumed_joules >= 0) & (df.totalexecutiontime_sec > 0)].copy()
    # Each DCGM row represents one (job,node,GPU) summary; avoid silently double counting.
    duplicates = df.duplicated(["id_job", "Node", "gpu_id"])
    if duplicates.any():
        raise ValueError(f"Found {int(duplicates.sum())} duplicate (job, node, GPU) rows. Investigate rather than summing twice.")
    grouped = df.groupby("id_job", sort=False)
    jobs = grouped.agg(
        gpu_count=("gpu_id", "size"),
        duration_sec=("totalexecutiontime_sec", "max"),
        sm_utilization_pct=("avgsmutilization_pct", "mean"),
        memory_utilization_pct=("avgmemoryutilization_pct", "mean"),
        peak_gpu_memory_gb=("maxgpumemoryused_bytes", "max"),
        pcie_rx_mb=("pcierxbandwidth_megabytes_avg", "mean"),
        pcie_tx_mb=("pcietxbandwidth_megabytes_avg", "mean"),
        measured_energy_joules=("energyconsumed_joules", "sum"),
    ).reset_index()
    jobs["peak_gpu_memory_gb"] /= 1024**3
    jobs["measured_energy_kwh"] = jobs["measured_energy_joules"] / JOULES_PER_KWH
    jobs = jobs.replace([np.inf, -np.inf], np.nan)
    return jobs


def score(y_true, y_pred) -> dict:
    predictions = np.maximum(0, np.asarray(y_pred, dtype=float))
    return {
        "mae_kwh": float(mean_absolute_error(y_true, predictions)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def run_experiment(data_path: str | Path, output_dir: str | Path, *, seed: int = 42) -> dict:
    jobs = load_jobs(data_path)
    if len(jobs) < 30:
        raise ValueError("At least 30 valid distinct jobs are needed for an evaluation split.")
    train, test = train_test_split(jobs, test_size=0.2, random_state=seed)
    x_train = train[FEATURES].copy()
    x_test = test[FEATURES].copy()
    # Fit imputation only on training data to prevent test-set leakage.
    median = x_train.median().fillna(0)
    x_train = x_train.fillna(median)
    x_test = x_test.fillna(median)
    y_train = train["measured_energy_kwh"]
    y_test = test["measured_energy_kwh"]
    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    # Log-transform handles the long-tailed distribution of GPU-job energy.
    model.fit(x_train, np.log1p(y_train))
    pred = np.expm1(model.predict(x_test))
    results = {
        "data_source": str(data_path),
        "job_count": int(len(jobs)),
        "train_jobs": int(len(train)),
        "test_jobs": int(len(test)),
        "train_test_split": "80/20, job-disjoint, random_state=" + str(seed),
        "features": FEATURES,
        "target": "sum energyconsumed_joules per job converted to kWh",
        "baseline_median": score(y_test, baseline.predict(x_test)),
        "hist_gradient_boosting": score(y_test, pred),
        "caution": "Predictive association only; not measured energy savings or causal optimisation.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame({"id_job": test.id_job.to_numpy(), "measured_kwh": y_test.to_numpy(), "predicted_kwh": np.maximum(pred, 0)}).to_csv(output / "holdout_predictions.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate job-level GPU energy prediction")
    parser.add_argument("--data", default="data/raw/dcgm.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
