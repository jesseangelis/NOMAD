"""Centralised SQLite gateway for the NOMAD pipeline.

This module is the *only* place in the codebase that may import ``sqlite3``.
All other modules read and write pipeline results exclusively through the
public functions defined here.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Yields a connection with WAL mode and foreign-key enforcement enabled."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Creates all tables if they do not exist. Never drops existing data.

    Args:
        db_path: Filesystem path to the SQLite database file.
    """
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS intensities (
                protein TEXT NOT NULL,
                gene_symbol TEXT,
                entry_name TEXT,
                description TEXT,
                sample  TEXT NOT NULL,
                intensity REAL,
                stderr    REAL,
                supported INTEGER,
                PRIMARY KEY (protein, sample)
            );
            CREATE TABLE IF NOT EXISTS emissions (
                protein   TEXT NOT NULL,
                precursor TEXT NOT NULL,
                probability REAL,
                PRIMARY KEY (protein, precursor)
            );
            CREATE TABLE IF NOT EXISTS diagnostic_loo (
                protein   TEXT NOT NULL,
                sample    TEXT NOT NULL,
                precursor TEXT NOT NULL,
                actual    REAL,
                predicted REAL,
                scale     REAL,
                PRIMARY KEY (protein, sample, precursor)
            );
        """)
    logger.debug("Database schema initialised at %s.", db_path)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def save_intensities(db_path: str, quant_df: pl.DataFrame) -> None:
    """Upserts the NMF W-matrix (intensities, stderr, supported) into the DB.

    Args:
        db_path: Path to the SQLite database.
        quant_df: Wide-format DataFrame with columns: protein, gene_symbol,
            entry_name, description, n_proteins, <sample>,
            <sample>_stderr, <sample>_supported.
    """
    sample_cols = [
        c for c in quant_df.columns
        if c not in ("protein", "gene_symbol", "entry_name", "description", "n_proteins", "structural_specificity")
        and not c.endswith("_stderr")
        and not c.endswith("_supported")
    ]
    rows = [
        (row["protein"], row.get("gene_symbol"), row.get("entry_name"), row.get("description"),
         s, float(row[s]),
         float(row.get(f"{s}_stderr", 0.0)),
         int(row.get(f"{s}_supported", True)))
        for row in quant_df.to_dicts()
        for s in sample_cols
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO intensities VALUES (?,?,?,?,?,?,?,?)", rows
        )
    logger.info("Saved %d intensity rows to %s.", len(rows), db_path)


def save_emissions(db_path: str, emissions_df: pl.DataFrame) -> None:
    """Upserts the NMF H-matrix (emission probabilities) into the DB.

    Args:
        db_path: Path to the SQLite database.
        emissions_df: DataFrame with columns: protein, precursor, probability.
    """
    rows = [
        (r["protein"], r["precursor"], float(r["probability"]))
        for r in emissions_df.to_dicts()
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO emissions VALUES (?,?,?)", rows
        )
    logger.info("Saved %d emission rows to %s.", len(rows), db_path)


def save_diagnostic_loo(
    db_path: str, cv_data: List[Tuple[float, float, float, str, str, str]]
) -> None:
    """Saves the Leave-One-Out (LOO) cross-validation diagnostics to the DB.

    Args:
        db_path: Path to the SQLite database.
        cv_data: List of (actual, predicted, scale, protein, sample, precursor) tuples.
    """
    rows = [
        (p[3], p[4], p[5], float(p[0]), float(p[1]), float(p[2]))
        for p in cv_data
        if len(p) >= 6
    ]
    if not rows:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO diagnostic_loo VALUES (?,?,?,?,?,?)", rows
        )
    logger.info("Saved %d diagnostic LOO rows to %s.", len(rows), db_path)
