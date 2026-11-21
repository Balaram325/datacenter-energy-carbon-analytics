"""Transparent hypothetical carbon-aware scheduling scenario.

This is NOT a validated real-world scheduling policy. We assume every selected
job can be deferred to cleaner electricity with identical energy consumption,
no deadlines, and unconstrained GPU capacity. Real deployments require those
constraints, electricity-time data, and measured outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis import load_jobs


def simulate(jobs, *, baseline_kg_per_kwh: float = 0.65,
             cleaner_kg_per_kwh: float = 0.30,
             flexible_energy_fraction: float = 0.20) -> dict:
    if not 0 <= flexible_energy_fraction <= 1:
        raise ValueError("flexible_energy_fraction must be between 0 and 1")
    if baseline_kg_per_kwh < 0 or cleaner_kg_per_kwh < 0:
        raise ValueError("Carbon intensity values must be nonnegative")
    if cleaner_kg_per_kwh > baseline_kg_per_kwh:
        raise ValueError("Cleaner intensity must not exceed baseline intensity")
    total_energy_kwh = float(jobs.measured_energy_kwh.sum())
    shifted_energy_kwh = total_energy_kwh * flexible_energy_fraction
    before = total_energy_kwh * baseline_kg_per_kwh
    after = (total_energy_kwh - shifted_energy_kwh) * baseline_kg_per_kwh + shifted_energy_kwh * cleaner_kg_per_kwh
    return {
        "scenario": "Idealised energy-fraction shift to lower-carbon grid hours (not implemented scheduling)",
        "jobs": int(len(jobs)),
        "total_measured_gpu_energy_kwh": total_energy_kwh,
        "baseline_grid_intensity_kgco2e_per_kwh_assumed": baseline_kg_per_kwh,
        "cleaner_grid_intensity_kgco2e_per_kwh_assumed": cleaner_kg_per_kwh,
        "fraction_of_energy_flexible_assumed": flexible_energy_fraction,
        "energy_shifted_kwh_assumed": shifted_energy_kwh,
        "baseline_emissions_kgco2e_scenario": before,
        "shifted_emissions_kgco2e_scenario": after,
        "avoided_emissions_kgco2e_scenario": before - after,
        "avoided_emissions_pct_scenario": (100 * (before - after) / before) if before else 0.0,
        "limitations": "The data contains no measured time-varying grid intensity or verified job flexibility. No evidence of operational CO2 reduction or guaranteed scheduling feasibility.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate hypothetical carbon-aware load-shift scenario")
    parser.add_argument("--data", default="data/raw/dcgm.csv")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--baseline-intensity", type=float, default=0.65)
    parser.add_argument("--cleaner-intensity", type=float, default=0.30)
    parser.add_argument("--flexible-fraction", type=float, default=0.20)
    args = parser.parse_args()
    result = simulate(load_jobs(args.data), baseline_kg_per_kwh=args.baseline_intensity,
                      cleaner_kg_per_kwh=args.cleaner_intensity,
                      flexible_energy_fraction=args.flexible_fraction)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "carbon_scenario.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
