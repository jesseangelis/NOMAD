"""Sparse averaging operators and replicate group indexing helpers."""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch


def prepare_averaging_matrices(
    metadata: pl.DataFrame, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Builds sparse matrices to average protein intensities by dose group.

    Used to enforce smooth dose-response monotonicity regularizers.
    """
    res: Dict[str, torch.Tensor] = {}
    if "name" not in metadata.columns or "dose" not in metadata.columns:
        return res

    drug_array = metadata["name"].to_numpy()
    doses_array = metadata["dose"].to_numpy()
    n_samples = len(metadata)

    for drug in metadata["name"].unique().to_list():
        drug_idx = np.where(drug_array == drug)[0]
        unique_doses = np.unique(doses_array[drug_idx])
        if len(unique_doses) < 2:
            continue

        rows, cols, data = [], [], []
        for i, dose in enumerate(unique_doses):
            idx = np.where((drug_array == drug) & (doses_array == dose))[0]
            if len(idx) > 0:
                val = 1.0 / len(idx)
                for j in idx:
                    rows.append(i)
                    cols.append(j)
                    data.append(val)

        mat = sp.csr_matrix((data, (rows, cols)), shape=(len(unique_doses), n_samples))
        res[drug] = torch.sparse_csr_tensor(
            torch.from_numpy(mat.indptr).to(torch.int64),
            torch.from_numpy(mat.indices).to(torch.int64),
            torch.from_numpy(mat.data).to(torch.float32),
            size=mat.shape,
            device=device,
        )

    return res


def prepare_replicate_groups(
    metadata: pl.DataFrame, device: torch.device
) -> List[torch.Tensor]:
    """Precomputes replicate index groups for the within-group variance penalty."""
    res: List[torch.Tensor] = []
    if "name" not in metadata.columns or "dose" not in metadata.columns:
        return res

    drug_array = metadata["name"].to_numpy()
    doses_array = metadata["dose"].to_numpy()

    for drug in metadata["name"].unique().to_list():
        drug_idx = np.where(drug_array == drug)[0]
        for dose in np.unique(doses_array[drug_idx]):
            idxs = np.where((drug_array == drug) & (doses_array == dose))[0]
            if len(idxs) > 1:
                res.append(torch.tensor(idxs, dtype=torch.long, device=device))

    return res
