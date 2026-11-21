import numpy as np
import pandas as pd
import pytest

from src.analysis import JOULES_PER_KWH, load_jobs, score
from src.optimise import simulate


def test_unit_conversion():
    assert JOULES_PER_KWH == 3_600_000


def test_aggregate_gpu_energy(tmp_path):
    table = pd.DataFrame([
        {"id_job": 1, "Node": "n1", "gpu_id": 0, "energyconsumed_joules": 3600000, "totalexecutiontime_sec": 300},
        {"id_job": 1, "Node": "n1", "gpu_id": 1, "energyconsumed_joules": 7200000, "totalexecutiontime_sec": 300},
        {"id_job": 2, "Node": "n1", "gpu_id": 0, "energyconsumed_joules": 1800000, "totalexecutiontime_sec": 150},
    ])
    path = tmp_path / "small.csv"
    table.to_csv(path, index=False)
    # Production CSV has additional telemetry; synthetic rows provide them as NaN.
    result = load_jobs(path).sort_values("id_job")
    assert result.gpu_count.tolist() == [2, 1]
    assert np.allclose(result.measured_energy_kwh, [3, 0.5])


def test_reject_duplicate_gpu_rows(tmp_path):
    path = tmp_path / "dupes.csv"
    pd.DataFrame([{ "id_job": 1,"Node": "n", "gpu_id": 0, "energyconsumed_joules": 10,"totalexecutiontime_sec": 5}] * 2).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_jobs(path)


def test_score_perfect():
    assert score([1, 2, 3], [1, 2, 3])["mae_kwh"] == 0


def test_hypothetical_carbon_scenario():
    jobs = pd.DataFrame({"measured_energy_kwh": [10.0, 20.0]})
    result = simulate(jobs, baseline_kg_per_kwh=0.6, cleaner_kg_per_kwh=0.2, flexible_energy_fraction=0.25)
    assert result["baseline_emissions_kgco2e_scenario"] == pytest.approx(18)
    assert result["avoided_emissions_kgco2e_scenario"] == pytest.approx(3)
    assert result["avoided_emissions_pct_scenario"] == pytest.approx(100 / 6)


@pytest.mark.parametrize("fraction", [-0.2, 1.1])
def test_bad_fraction(fraction):
    with pytest.raises(ValueError):
        simulate(pd.DataFrame({"measured_energy_kwh": [1]}), flexible_energy_fraction=fraction)


def test_bad_intensity():
    with pytest.raises(ValueError):
        simulate(pd.DataFrame({"measured_energy_kwh": [1]}), baseline_kg_per_kwh=0.2, cleaner_kg_per_kwh=0.5)
