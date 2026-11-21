"""Plot holdout predictions produced by the modelling command."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/holdout_predictions.csv")
    parser.add_argument("--output", default="reports/holdout_scatter.png")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data.measured_kwh, data.predicted_kwh, s=5, alpha=0.25)
    limit = max(data.measured_kwh.max(), data.predicted_kwh.max())
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
    ax.set(xlabel="Measured GPU job energy (kWh)", ylabel="Predicted GPU job energy (kWh)", title="Holdout: predicted versus measured energy")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
