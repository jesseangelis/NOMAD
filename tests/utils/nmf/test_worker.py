"""Unit tests for nomad.utils.nmf.worker."""

import polars as pl
import pytest


@pytest.fixture
def minimal_meta():
    """Provides minimal metadata for worker initialisation tests."""
    return pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })


@pytest.mark.unit
def test_run_worker_batch_without_init_returns_empty(mocker):
    """Verifies that run_worker_batch returns empty lists when no engine is initialised."""
    from nomad.utils.nmf import worker

    # Patch the module-level engine to None (no init called)
    mocker.patch.object(worker, "_worker_engine", None)

    q_list, e_list, n, cv_list, rep_list, rs_list = worker.run_worker_batch(
        [{"component": "dummy"}]
    )

    assert q_list == []
    assert e_list == []
    assert n == 1
    assert cv_list == []


@pytest.mark.unit
def test_run_worker_batch_delegates_to_engine(mocker, minimal_meta):
    """Verifies that run_worker_batch calls _fit_subset on the engine and returns its output."""
    from nomad.utils.nmf import worker

    mock_engine = mocker.MagicMock()
    mock_engine._fit_subset.return_value = (
        [{"protein": "P1"}],  # q_list
        [{"protein": "P1", "precursor": "Pre1", "probability": 0.9}],  # e_list
        [],  # cv_list
        [],  # rep_list
        [],  # rs_list
    )
    mocker.patch.object(worker, "_worker_engine", mock_engine)

    q_list, e_list, n, cv_list, rep_list, rs_list = worker.run_worker_batch(
        ["component_A", "component_B"]
    )

    mock_engine._fit_subset.assert_called_once_with(
        ["component_A", "component_B"], show_progress=False
    )
    assert len(q_list) == 1
    assert n == 2


@pytest.mark.unit
def test_run_worker_batch_handles_runtime_error_gracefully(mocker):
    """Verifies that a RuntimeError in the engine returns empty results rather than raising."""
    from nomad.utils.nmf import worker

    mock_engine = mocker.MagicMock()
    mock_engine._fit_subset.side_effect = RuntimeError("GPU exploded")
    mocker.patch.object(worker, "_worker_engine", mock_engine)

    q_list, e_list, n, cv_list, rep_list, rs_list = worker.run_worker_batch(["c1"])

    assert q_list == []
    assert e_list == []
    assert n == 1
