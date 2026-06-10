"""Read and export functions for the NOMAD SQLite database."""

from __future__ import annotations

import logging
from typing import List, Optional

import polars as pl

from nomad.utils.db.db_write import _connect

logger = logging.getLogger(__name__)


def load_intensities(db_path: str, proteins: Optional[List[str]] = None) -> pl.DataFrame:
    """Loads the NMF W-matrix from the database.

    Args:
        db_path: Path to the SQLite database.
        proteins: Optional list of protein IDs to filter by.

    Returns:
        A wide-format DataFrame matching the output of NMFFit.
    """
    query = "SELECT * FROM intensities"
    params = []
    if proteins:
        placeholders = ",".join("?" * len(proteins))
        query += f" WHERE protein IN ({placeholders})"
        params.extend(proteins)

    ov = {"intensity": pl.Float64, "stderr": pl.Float64, "supported": pl.Int64}
    with _connect(db_path) as conn:
        df = pl.read_database(query, conn, execute_options={"parameters": params}, schema_overrides=ov)

    if df.is_empty():
        return df

    samples = sorted(df["sample"].unique().to_list())
    
    # Pivot each value type using multiple index columns
    idx_cols = ["protein", "gene_symbol", "entry_name", "description"]
    
    wide_df = df.pivot(values="intensity", index=idx_cols, on="sample", aggregate_function="first")
    stderr_df = df.pivot(values="stderr", index=idx_cols, on="sample", aggregate_function="first").rename({s: f"{s}_stderr" for s in samples})
    supp_df = df.pivot(values="supported", index=idx_cols, on="sample", aggregate_function="first").rename({s: f"{s}_supported" for s in samples})
    
    return wide_df.join(stderr_df, on=idx_cols).join(supp_df, on=idx_cols)


def load_emissions(db_path: str, proteins: Optional[List[str]] = None) -> pl.DataFrame:
    """Loads the NMF H-matrix from the database.

    Args:
        db_path: Path to the SQLite database.
        proteins: Optional list of protein IDs to filter by.

    Returns:
        A DataFrame with columns: protein, precursor, probability.
    """
    query = "SELECT * FROM emissions"
    params = []
    if proteins:
        placeholders = ",".join("?" * len(proteins))
        query += f" WHERE protein IN ({placeholders})"
        params.extend(proteins)

    with _connect(db_path) as conn:
        return pl.read_database(query, conn, execute_options={"parameters": params}, schema_overrides={"probability": pl.Float64})


def load_raw_intensities(db_path: str, proteins: Optional[List[str]] = None) -> pl.DataFrame:
    """Loads the long-format (raw) intensities from the database.

    Args:
        db_path: Path to the SQLite database.
        proteins: Optional list of protein IDs to filter by.

    Returns:
        A DataFrame with columns: protein, gene_symbol, entry_name, description, sample, intensity, stderr, supported.
    """
    query = "SELECT * FROM intensities"
    params = []
    if proteins:
        placeholders = ",".join("?" * len(proteins))
        query += f" WHERE protein IN ({placeholders})"
        params.extend(proteins)

    ov = {"intensity": pl.Float64, "stderr": pl.Float64, "supported": pl.Int64}
    with _connect(db_path) as conn:
        return pl.read_database(query, conn, execute_options={"parameters": params}, schema_overrides=ov)


def load_dose_response(db_path: str) -> pl.DataFrame:
    """Loads dose-response fitting results from the database.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        A DataFrame with columns matching the expected structure of VolcanoPlot.
    """
    query = "SELECT * FROM dose_response"
    with _connect(db_path) as conn:
        df = pl.read_database(query, conn)
    # Rename/alias columns so they match VolcanoPlot input expected structure
    if not df.is_empty():
        df = df.rename({
            "log2fc": "log2fc",
            "relevance_score": "relevance_score",
            "p_val": "p_val"
        })
    return df

