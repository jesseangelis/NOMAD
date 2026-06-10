"""Constants used across the NOMAD package."""

from __future__ import annotations

from enum import Enum


class GraphRelation(str, Enum):
    """Edge relation types used in the NOMAD protein-peptide-precursor graph."""

    PRODUCES = "PRODUCES"
    HAS_PRECURSOR = "HAS_PRECURSOR"
    DETECTED_IN = "DETECTED_IN"


class NodeType(str, Enum):
    """Node type labels used in the NOMAD graph."""

    PROTEIN = "Protein"
    PEPTIDE = "Peptide"
    PRECURSOR = "Precursor"
    SAMPLE = "Sample"


#: Minimum number of sample detections required for a precursor to be retained
#: during graph pruning.
MIN_PRECURSOR_DETECTIONS: int = 3

#: Supported MS quantification engine identifiers.
SUPPORTED_ENGINES: frozenset[str] = frozenset(
    {"fragpipe_combined"}
)

ENZYME_REGEX: dict[str, str] = {
    "trypsin/p": r"(?<=[KR])",
    "trypsin": r"(?<=[KR])(?!P)",
    "lys_c": r"(?<=K)",
    "lys_n": r"(?=K)",
    "glu_c": r"(?<=E)",
    "asp_n": r"(?=D)",
    "chymotrypsin": r"(?<=[FYWL])(?!P)",
}