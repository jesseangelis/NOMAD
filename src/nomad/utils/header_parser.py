"""Utilities for parsing UniProt and generic FASTA header strings."""

from __future__ import annotations

import re
from typing import Tuple


def parse_uniprot_header(header: str) -> Tuple[str, str]:
    """Parses a UniProt FASTA header to extract the accession and entry name.

    Handles standard SwissProt (sp|) and TrEMBL (tr|) prefixes as well as
    generic FASTA headers that do not follow the UniProt convention.

    Args:
        header: A raw FASTA header string, with or without a leading '>'.

    Returns:
        A tuple of (protein_id, entry_name). For non-UniProt headers the
        full stripped header is returned as protein_id and entry_name is
        an empty string.

    Examples:
        >>> parse_uniprot_header(">sp|P12345|ALBU_HUMAN Serum albumin")
        ('P12345', 'ALBU_HUMAN')
        >>> parse_uniprot_header(">generic_protein")
        ('generic_protein', '')
    """
    if not header or not isinstance(header, str):
        return str(header), ""

    stripped = header.lstrip(">").strip()

    if "|" in stripped:
        parts = stripped.split("|")
        if len(parts) >= 3:
            protein_id = parts[1].strip()
            # Entry name is the first token of the third field
            entry_name = parts[2].split()[0].strip() if parts[2] else ""
            return protein_id, entry_name

    # Generic header: return the first whitespace-delimited token
    return (stripped.split()[0], "") if stripped else ("", "")


def parse_header_details(header: str) -> Tuple[str, str, str, str]:
    """Parses a UniProt/generic FASTA header string.

    Args:
        header: Raw header line (with or without '>').

    Returns:
        A tuple of (protein_id, entry_name, gene_symbol, description)
    """
    if not header or not isinstance(header, str):
        return "", "", "", ""

    stripped = header.lstrip(">").strip()

    if "|" in stripped:
        parts = stripped.split("|")
        if len(parts) >= 3:
            protein_id = parts[1].strip()
            rest = parts[2].strip()

            # Entry name is the first whitespace-delimited token of the third field
            rest_parts = rest.split(None, 1)
            entry_name = rest_parts[0] if len(rest_parts) > 0 else ""
            remaining = rest_parts[1] if len(rest_parts) > 1 else ""

            # Description is the text in remaining up to any tag like OS=, OX=, GN=, PE=, SV=
            description = remaining
            for tag in ["OS=", "OX=", "GN=", "PE=", "SV="]:
                if tag in description:
                    description = description.split(tag)[0].strip()

            # Gene symbol
            gene_symbol = ""
            gn_match = re.search(r"\bGN=([^\s]+)", remaining)
            if gn_match:
                gene_symbol = gn_match.group(1)

            return protein_id, entry_name, gene_symbol, description

    # Generic header
    tokens = stripped.split(None, 1)
    protein_id = tokens[0] if tokens else ""
    description = tokens[1] if len(tokens) > 1 else ""
    return protein_id, "", "", description


def parse_group_header_details(group_header: str) -> Tuple[str, str, str, str]:
    """Parses a combined group header of proteins (separated by '; ').

    Returns:
        A tuple of:
        - accessions: '; ' joined protein IDs
        - entry_names: '; ' joined entry names
        - genes: '; ' joined unique gene symbols
        - descriptions: '; ' joined unique descriptions
    """
    ids = []
    entries = []
    genes = []
    descs = []

    for h in group_header.split("; "):
        h = h.strip()
        if not h:
            continue
        p_id, entry, gene, desc = parse_header_details(h)
        if p_id:
            ids.append(p_id)
        if entry:
            entries.append(entry)
        if gene:
            genes.append(gene)
        if desc:
            descs.append(desc)

    def uniq(lst):
        seen = set()
        res = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                res.append(x)
        return res

    return (
        "; ".join(uniq(ids)),
        "; ".join(uniq(entries)),
        "; ".join(uniq(genes)),
        "; ".join(uniq(descs))
    )
