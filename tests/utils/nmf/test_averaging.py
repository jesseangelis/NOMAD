"""Unit tests for nomad.utils.nmf.averaging."""

import polars as pl
import pytest
import torch

from nomad.utils.nmf.averaging import prepare_averaging_matrices, prepare_replicate_groups


@pytest.fixture
def multi_dose_metadata():
    """Provides metadata with two drugs and three replicates per dose."""
    return pl.DataFrame({
        "name": ["DrugA"] * 9,
        "dose": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0],
        "file": [f"S{i}" for i in range(9)],
    })


@pytest.fixture
def device():
    """Provides the default torch device for testing."""
    return torch.device("cpu")


@pytest.mark.unit
def test_prepare_averaging_matrices_empty_metadata(device):
    """Verifies that empty metadata produces an empty averaging matrix dict."""
    result = prepare_averaging_matrices(pl.DataFrame(), device)

    assert result == {}


@pytest.mark.unit
def test_prepare_averaging_matrices_missing_dose_column(device):
    """Verifies that metadata without a dose column produces an empty dict."""
    meta = pl.DataFrame({"name": ["DrugA"], "file": ["S1"]})
    result = prepare_averaging_matrices(meta, device)

    assert result == {}


@pytest.mark.unit
def test_prepare_averaging_matrices_single_dose_excluded(device):
    """Verifies that a drug with only one dose level is excluded (no smoothing needed)."""
    meta = pl.DataFrame({
        "name": ["DrugA", "DrugA"],
        "dose": [1.0, 1.0],
        "file": ["S1", "S2"],
    })
    result = prepare_averaging_matrices(meta, device)

    assert "DrugA" not in result


@pytest.mark.unit
def test_prepare_averaging_matrices_multi_dose_shape(multi_dose_metadata, device):
    """Verifies that a multi-dose drug produces a matrix with shape (n_doses, n_samples)."""
    result = prepare_averaging_matrices(multi_dose_metadata, device)

    assert "DrugA" in result
    A = result["DrugA"]
    n_samples = len(multi_dose_metadata)
    n_doses = 3  # 1.0, 10.0, 100.0
    assert A.shape == (n_doses, n_samples)


@pytest.mark.unit
def test_prepare_averaging_matrices_row_sums_to_one(multi_dose_metadata, device):
    """Verifies that each row of the averaging matrix sums to approximately 1.0."""
    result = prepare_averaging_matrices(multi_dose_metadata, device)
    A = result["DrugA"].to_dense()

    row_sums = A.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


@pytest.mark.unit
def test_prepare_replicate_groups_empty_metadata(device):
    """Verifies that empty metadata produces an empty replicate group list."""
    result = prepare_replicate_groups(pl.DataFrame(), device)

    assert result == []


@pytest.mark.unit
def test_prepare_replicate_groups_no_replicates(device):
    """Verifies that a dataset with one sample per dose produces no replicate groups."""
    meta = pl.DataFrame({
        "name": ["DrugA", "DrugA", "DrugA"],
        "dose": [1.0, 10.0, 100.0],
        "file": ["S1", "S2", "S3"],
    })
    result = prepare_replicate_groups(meta, device)

    assert result == []


@pytest.mark.unit
def test_prepare_replicate_groups_with_replicates(multi_dose_metadata, device):
    """Verifies that three-replicate doses produce three replicate index tensors."""
    result = prepare_replicate_groups(multi_dose_metadata, device)

    # Three doses × one drug = three groups
    assert len(result) == 3
    for group in result:
        assert isinstance(group, torch.Tensor)
        assert group.shape[0] == 3  # Three replicates per dose
