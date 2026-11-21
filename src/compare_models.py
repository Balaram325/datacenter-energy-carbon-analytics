"""Compare linear and boosted energy models; inspect permutation importance.

Uses the original DCGM data locally and the SAME job-disjoint split as src.analysis.
Run: python -m src.compare_models --data data/raw/dcgm.csv --output reports
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import FEATURES, load_jobs, score


def compare_models(data_path, output_dir, seed=42, importance_sample=2500, repeats=3):
    jobs = load_jobs(data_path)
    if len(jobs) < 30:
        raise ValueError("At least 30 distinct jobs are required")
    train, test = train_test_split(jobs, test_size=0.2, random_state=seed)
    train_x, test_x = train[FEATURES], test[FEATURES]
    medians = train_x.median().fillna(0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    train_y, test_y = train.measured_energy_kwh, test.measured_energy_kwh
    median_model = DummyRegressor(strategy="median").fit(train_x, train_y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train_x, np.log1p(train_y))
    boosted = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                            l2_regularization=1.0, random_state=seed)
    boosted.fit(train_x, np.log1p(train_y))
    baseline_pred = median_model.predict(test_x)
    ridge_pred = np.expm1(ridge.predict(test_x))
    boosted_pred = np.expm1(boosted.predict(test_x))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(test_x), size=min(len(test_x), importance_sample), replace=False)
    subset_x = test_x.iloc[chosen].copy()
    subset_y = test_y.iloc[chosen]
    baseline_error = mean_absolute_error(subset_y, np.maximum(0, np.expm1(boosted.predict(subset_x))))
    importances = {}
    for feature in FEATURES:
        increases = []
        for _ in range(repeats):
            modified = subset_x.copy()
            modified[feature] = rng.permutation(modified[feature].to_numpy())
            pred = np.maximum(0, np.expm1(boosted.predict(modified)))
            increases.append(float(mean_absolute_error(subset_y, pred) - baseline_error))
        importances[feature] = float(np.mean(increases))

    result = {
        "jobs": len(jobs), "test_jobs": len(test),
        "split": f"80/20 job-disjoint, random_state={seed}",
        "median_baseline": score(test_y, baseline_pred),
        "ridge_log_target": score(test_y, ridge_pred),
        "hist_gradient_boosting_log_target": score(test_y, boosted_pred),
        "permutation_importance": {
            "method": f"Mean MAE increase on {len(chosen)} held-out jobs, {repeats} shuffles per feature",
            "features": dict(sorted(importances.items(), key=lambda x: -x[1]))
        },
        "caution": "Retrospective job-energy estimation; completed-job duration is a feature. "
                   "Permutation importances are associations and can be affected by correlated features."
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare energy regressors and permutation importance")
    parser.add_argument("--data", default="data/raw/dcgm.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    print(json.dumps(compare_models(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
