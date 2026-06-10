
import pytest
import numpy as np
import polars as pl
from nomad.utils.nmf import NMFFit

def test_init(graph_simple):
    meta = pl.DataFrame({"file": ["S1", "S2"], "name": ["DrugA", "DrugA"], "dose": [0.0, 10.0]})
    fitter = NMFFit(graph_simple, meta_df=meta)
    assert "S1" in fitter.samples
    assert "S2" in fitter.samples
    assert fitter.quant_df.is_empty()
    assert fitter.lambda_reg == 1e-3
    assert fitter.gamma_dr == 10.0
    assert fitter.beta_rep == 0.1

def test_global_matrix_construction(graph_simple):
    """Test component matrix construction."""
    from nomad.utils import graph_ops
    precursors = ["Pre1", "Pre2", "Pre3"]
    proteins = ["P1", "P2"]
    s2i = {"S1": 0, "S2": 1}
    
    v_mat = graph_ops.build_v_matrix(graph_simple, precursors, s2i)
    assert v_mat.shape == (2, 3)
    assert v_mat[0, 0] == 100.0
    assert v_mat[1, 2] == 0.0 # Pre3 missing in S2
    
    h_mask = graph_ops.build_h_mask(graph_simple, proteins, precursors)
    assert h_mask.shape == (2, 3)
    assert h_mask[0, 0] == 1.0 # P1 -> Pep1 -> Pre1
    assert h_mask[0, 2] == 0.0 # P1 does not produce Pre3

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_grouping(graph_simple):
    # Add a protein P3 that produces exactly same peptides as P1
    graph_simple.add_node("P3", type="Protein")
    graph_simple.add_edge("P3", "Pep1", relation="PRODUCES")
    graph_simple.add_edge("P3", "Pep2", relation="PRODUCES")
    
    meta = pl.DataFrame({"file": ["S1", "S2"], "name": ["DrugA", "DrugA"], "dose": [0.0, 10.0]})
    fitter = NMFFit(graph_simple, meta_df=meta)
    res_df, stats = fitter.fit()
    
    assert res_df.height == 2
    
    # P1 and P3 should be grouped together
    grouped_row = res_df.filter(pl.col("protein").str.contains("P1") & pl.col("protein").str.contains("P3"))
    assert not grouped_row.is_empty()
    assert grouped_row.height == 1
    assert grouped_row["S1"][0] > 0


def test_prior_penalty_masking(graph_simple):
    """Test that the Specific-Precursor Prior Penalty correctly suppresses isoform abundance when specific precursor is absent."""
    meta = pl.DataFrame({"file": ["S1", "S2"], "name": ["DrugA", "DrugA"], "dose": [0.0, 10.0]})
    
    # Fit 1: Standard NMF (gamma_prior = 0.0)
    fitter_std = NMFFit(graph_simple, meta_df=meta, gamma_prior=0.0)
    res_std, _ = fitter_std.fit()
    
    # Get standard abundance of P2 in S2
    p2_row_std = res_std.filter(pl.col("protein") == "P2")
    assert not p2_row_std.is_empty()
    p2_s2_std = p2_row_std["S2"][0]
    
    # Fit 2: Prior Penalty enabled (gamma_prior = 100.0 to ensure strong suppression in simple mock data)
    fitter_prior = NMFFit(graph_simple, meta_df=meta, gamma_prior=100.0)
    res_prior, _ = fitter_prior.fit()
    
    p2_row_prior = res_prior.filter(pl.col("protein") == "P2")
    assert not p2_row_prior.is_empty()
    p2_s2_prior = p2_row_prior["S2"][0]
    
    # Since Pre3 (P2's only specific precursor) is missing in S2, the prior penalty should suppress P2's abundance in S2 to near zero.
    assert p2_s2_prior < 1e-4
    assert p2_s2_prior < p2_s2_std

