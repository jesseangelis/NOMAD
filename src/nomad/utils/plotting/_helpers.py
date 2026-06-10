"""Internal helper utilities standardizing protein headers and graphical styling tokens."""

from __future__ import annotations

from typing import List
import polars as pl

from nomad.utils.header_parser import parse_uniprot_header

# Corporate visual palette for consistent isoform rendering
PREMIUM_COLORS: List[str] = [
    "#E74C3C",
    "#2ECC71",
    "#3498DB",
    "#F1C40F",
    "#9B59B6",
    "#1ABC9C",
]


def deduplicate_ids(col: pl.Expr) -> pl.Expr:
    """Polars expression to deduplicate semicolon-separated identifier lists."""
    return col.map_elements(
        lambda x: "; ".join(dict.fromkeys(str(x).replace(" ", "").split(";")))
        if x
        else x,
        return_dtype=pl.Utf8,
    )


def clean_prot(n: str) -> str:
    """Truncates protein strings to standard visible identifiers for layout fit."""
    headers = n.split("; ")
    ids = []
    for h in headers:
        p_id, _ = parse_uniprot_header(h)
        ids.append(p_id if "|" in h else h[:15])
    if len(ids) > 2:
        return "; ".join(ids[:2]) + "; ..."
    return "; ".join(ids)
