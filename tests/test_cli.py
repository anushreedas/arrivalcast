import json

import pandas as pd
import pytest

from arrivalcast import cli
from arrivalcast.data import pipeline
from arrivalcast.data.io import load_sources


def test_loader_returns_sorted_tz_naive_series(tiny_dataset):
    src = load_sources(str(tiny_dataset))
    assert set(src) == {"SRC_A", "SRC_B"}
    ts = src["SRC_A"]["arrival_timestamp"]
    assert ts.is_monotonic_increasing and ts.dt.tz is None


def test_synthetic_generation_is_deterministic_per_source():
    from datetime import datetime
    a = pipeline.generate_synthetic_source("SRC_02", datetime(2023, 1, 1), datetime(2023, 3, 1))
    b = pipeline.generate_synthetic_source("SRC_02", datetime(2023, 1, 1), datetime(2023, 3, 1))
    pd.testing.assert_frame_equal(a, b)
    assert len(pipeline.SYNTHETIC_REGISTRY) == 26


def test_bts_requires_frozen_settings(tiny_dataset):
    with pytest.raises(SystemExit, match="held-out"):
        cli.main(["run", "--dataset", "bts", "--synthetic-path", str(tiny_dataset)])


@pytest.mark.slow
def test_end_to_end_run_and_predict(tmp_path, tiny_dataset, capsys):
    out = tmp_path / "results"
    args = ["run", "--out-dir", str(out), "--run-name", "smoke", "--synthetic-path", str(tiny_dataset),
            "--max-epochs", "2", "--patience", "1", "--hidden-dim", "8", "--n-components", "2"]
    assert cli.main(args) == 0
    run = out / "smoke"
    metrics = pd.read_csv(run / "full_metric_evaluation.csv")
    assert set(metrics.model) == {"Poisson", "Hawkes", "NeuralHawkes", "LogNormMix"}
    assert metrics.mean_nll.notna().all()
    assert (run / "test_predictions.parquet").exists()
    # re-running with different settings under the same name is refused
    with pytest.raises(SystemExit, match="different training settings"):
        cli.main(args[:-1] + ["3"])
    # a frozen BTS-style run may not be combined with training flags
    with pytest.raises(SystemExit, match="cannot also be given"):
        cli.main(["run", "--freeze-from", str(run), "--lr", "0.1", "--synthetic-path", str(tiny_dataset)])

    capsys.readouterr()
    ckpt = run / "models" / "SRC_A__lnm_s0.pt"
    assert cli.main(["predict", "--checkpoint", str(ckpt), "--history", str(tiny_dataset / "SRC_A.csv"),
                     "--sla-hours", "24"]) == 0
    fc = json.loads(capsys.readouterr().out)
    assert 0 <= fc["p_within_sla"] <= 1 and len(fc["quantiles_hours"]) == 7
