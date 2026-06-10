"""Unit tests for nomad.utils.constants."""

import pytest

from nomad.utils.constants import (
    ENZYME_REGEX,
    MIN_PRECURSOR_DETECTIONS,
    SUPPORTED_ENGINES,
    GraphRelation,
    NodeType,
)


@pytest.mark.unit
def test_graph_relation_values():
    """Verifies that GraphRelation enum members have the expected string values."""
    assert GraphRelation.PRODUCES == "PRODUCES"
    assert GraphRelation.HAS_PRECURSOR == "HAS_PRECURSOR"
    assert GraphRelation.DETECTED_IN == "DETECTED_IN"


@pytest.mark.unit
def test_graph_relation_is_string_subclass():
    """Verifies that GraphRelation members behave as plain strings."""
    assert isinstance(GraphRelation.PRODUCES, str)


@pytest.mark.unit
def test_node_type_values():
    """Verifies that NodeType enum members have the expected string values."""
    assert NodeType.PROTEIN == "Protein"
    assert NodeType.PEPTIDE == "Peptide"
    assert NodeType.PRECURSOR == "Precursor"
    assert NodeType.SAMPLE == "Sample"


@pytest.mark.unit
def test_node_type_is_string_subclass():
    """Verifies that NodeType members behave as plain strings."""
    assert isinstance(NodeType.SAMPLE, str)


@pytest.mark.unit
def test_min_precursor_detections_value():
    """Verifies that MIN_PRECURSOR_DETECTIONS equals 3."""
    assert MIN_PRECURSOR_DETECTIONS == 3


@pytest.mark.unit
def test_supported_engines_contains_fragpipe():
    """Verifies that fragpipe_combined is in SUPPORTED_ENGINES."""
    assert "fragpipe_combined" in SUPPORTED_ENGINES


@pytest.mark.unit
def test_supported_engines_is_immutable():
    """Verifies that SUPPORTED_ENGINES is a frozenset (immutable)."""
    assert isinstance(SUPPORTED_ENGINES, frozenset)


@pytest.mark.unit
@pytest.mark.parametrize("enzyme", [
    "trypsin",
    "trypsin/p",
    "lys_c",
    "lys_n",
    "glu_c",
    "asp_n",
    "chymotrypsin",
])
def test_enzyme_regex_contains_all_supported_enzymes(enzyme):
    """Verifies that ENZYME_REGEX contains an entry for each supported enzyme."""
    assert enzyme in ENZYME_REGEX
    assert isinstance(ENZYME_REGEX[enzyme], str)
    assert len(ENZYME_REGEX[enzyme]) > 0
