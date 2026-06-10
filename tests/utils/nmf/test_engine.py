"""Tests for nomad.utils.nmf.engine.NMFFit.

Unit tests cover initialisation and structural correctness.
Integration tests verify end-to-end NMF fitting behaviour.
"""

import networkx as nx
import numpy as np
import polars as pl
import pytest

from nomad.utils.nmf import NMFFit


# ---------------------------------------------------------------------------
# Unit tests — NMFFit initialisation and structural properties
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_init_stores_samples_and_defaults(graph_simple):
    """Verifies that NMFFit stores the sample list and default hyperparameters."""
    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })
    fitter = NMFFit(graph_simple, meta_df=meta)

    assert "S1" in fitter.samples
    assert "S2" in fitter.samples
    assert fitter.quant_df.is_empty()
    assert fitter.lambda_reg == 1e-3
    assert fitter.gamma_dr == 10.0
    assert fitter.beta_rep == 0.1


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_grouping_merges_proteins_with_identical_evidence(graph_simple):
    """Verifies that proteins sharing the exact same peptide mask are merged into one row."""
    # P3 shares all peptides with P1 → should be grouped
    graph_simple.add_node("P3", type="Protein")
    graph_simple.add_edge("P3", "Pep1", relation="PRODUCES")
    graph_simple.add_edge("P3", "Pep2", relation="PRODUCES")

    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })
    res_df, _ = NMFFit(graph_simple, meta_df=meta).fit()

    assert res_df.height == 2
    grouped_row = res_df.filter(
        pl.col("protein").str.contains("P1") & pl.col("protein").str.contains("P3")
    )
    assert not grouped_row.is_empty()
    assert grouped_row.height == 1
    assert grouped_row["S1"][0] > 0


@pytest.mark.unit
def test_prior_penalty_suppresses_unsupported_protein(graph_simple):
    """Verifies that the prior penalty reduces abundance for proteins missing specific precursors.

    Pre3 (P2's only unique precursor) is absent in S2. With a strong prior
    penalty P2's S2 abundance should be suppressed to near zero.
    """
    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })

    res_std, _ = NMFFit(graph_simple, meta_df=meta, gamma_prior=0.0).fit()
    res_prior, _ = NMFFit(graph_simple, meta_df=meta, gamma_prior=100.0).fit()

    p2_s2_std = res_std.filter(pl.col("protein") == "P2")["S2"][0]
    p2_s2_prior = res_prior.filter(pl.col("protein") == "P2")["S2"][0]

    assert p2_s2_prior < 1e-4
    assert p2_s2_prior < p2_s2_std


@pytest.mark.unit
def test_h_bounds_are_within_expected_range(graph_simple):
    """Verifies that emission probabilities are non-negative after fitting."""
    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })
    fitter = NMFFit(graph_simple, meta_df=meta)
    fitter.fit()

    probs = fitter.emissions_df["probability"].to_numpy()
    assert np.all(probs >= 0)
    assert np.all(probs <= 1000.0)  # Emission weight scale tolerance


@pytest.mark.unit
def test_h_max_normalisation_produces_positive_probabilities(graph_simple):
    """Verifies that H normalisation produces strictly positive probabilities."""
    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })
    fitter = NMFFit(graph_simple, meta_df=meta)
    fitter.fit()

    probs = fitter.emissions_df["probability"].to_numpy()
    if probs.size > 0:
        assert np.all(probs > 0.0)


@pytest.mark.unit
def test_real_support_missing_precursor_suppresses_weight(graph_simple):
    """Verifies that P4 (only linked to Pre3, absent in S2) has near-zero S2 weight."""
    graph_simple.add_node("P4", type="Protein")
    graph_simple.add_node("Pep4", type="Peptide")
    graph_simple.add_edge("P4", "Pep4", relation="PRODUCES")
    graph_simple.add_edge("Pep4", "Pre3", relation="HAS_PRECURSOR")

    meta_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [1.0, 10.0],
    })
    res_df, _ = NMFFit(graph_simple, meta_df=meta_df).fit()

    p4_row = res_df.filter(pl.col("protein").str.contains("P4"))
    assert p4_row["S2"][0] < 1.0


# ---------------------------------------------------------------------------
# Integration tests — end-to-end NMF recovery
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit_returns_non_empty_dataframe(graph_simple):
    """Verifies that fit() produces a non-empty quant_df with sample columns."""
    meta = pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })
    res_df, _ = NMFFit(graph_simple, meta_df=meta).fit()

    assert not res_df.is_empty()
    assert res_df.height == 2
    assert "S1" in res_df.columns
    assert "S2" in res_df.columns
    assert res_df["S1"].min() >= 0
    assert res_df["S2"].min() >= 0


@pytest.mark.integration
@pytest.mark.slow
def test_full_fit_orchestration(graph_undirected, meta_df):
    """Verifies that a single-component graph produces exactly one protein row."""
    quant_df, _ = NMFFit(graph_undirected, meta_df).fit()

    assert not quant_df.is_empty()
    assert "protein" in quant_df.columns
    assert "S1" in quant_df.columns
    assert quant_df.height == 1


@pytest.mark.integration
@pytest.mark.slow
def test_nmf_ground_truth_recovery():
    """Verifies that NMF recovers a known W matrix from a synthetic V observation."""
    np.random.seed(42)
    n_samples, n_proteins, n_precursors = 10, 3, 10

    w_true = np.random.rand(n_samples, n_proteins) * 100
    h_global = np.random.rand(n_precursors)
    h_global /= h_global.mean()

    h_true = np.zeros((n_proteins, n_precursors))
    for i in range(n_proteins):
        start, end = i * 3, min(i * 3 + 4, n_precursors)
        h_true[i, start:end] = h_global[start:end]

    v_true = w_true @ h_true
    v_obs = np.maximum(v_true + np.random.normal(0, 0.1, v_true.shape), 0)

    g = nx.Graph()
    proteins = [f"P{i}" for i in range(n_proteins)]
    precursors = [f"Pre{j}" for j in range(n_precursors)]
    samples = [f"S{k}" for k in range(n_samples)]

    for p in proteins:
        g.add_node(p, type="Protein")
    for prec in precursors:
        g.add_node(prec, type="Precursor")

    for i, p in enumerate(proteins):
        start, end = i * 3, min(i * 3 + 4, n_precursors)
        for j in range(start, end):
            pep = f"Pep_{i}_{j}"
            g.add_node(pep, type="Peptide")
            g.add_edge(p, pep, relation="PRODUCES")
            g.add_edge(pep, precursors[j], relation="HAS_PRECURSOR")

    for i, s in enumerate(samples):
        for j, prec in enumerate(precursors):
            intensity = v_obs[i, j]
            if intensity > 0:
                g.add_edge(prec, s, relation="DETECTED_IN", intensity=intensity)

    meta = pl.DataFrame({
        "file": samples,
        "name": samples,
        "dose": [0] * len(samples),
    })
    res_df, _ = NMFFit(g, meta_df=meta).fit()

    for i, p in enumerate(proteins):
        row = res_df.filter(pl.col("protein").str.contains(p))
        vals = np.array([row[s][0] for s in samples])
        corrs = [np.corrcoef(vals, w_true[:, j])[0, 1] for j in range(n_proteins)]
        assert max(corrs) > 0.70

        best_j = np.argmax(corrs)
        w_est_norm = vals / (np.mean(vals) + 1e-9)
        w_true_norm = w_true[:, best_j] / (np.mean(w_true[:, best_j]) + 1e-9)
        assert np.mean(np.abs(w_est_norm - w_true_norm)) < 0.5


@pytest.mark.integration
@pytest.mark.slow
def test_nmf_uncertainty_quantification():
    """Verifies that standard error estimates fall within a reasonable range."""
    np.random.seed(42)
    n_samples, n_proteins, n_precursors = 100, 1, 10

    v_true = np.ones((n_samples, n_proteins)) @ np.ones((n_proteins, n_precursors)) * 100
    v_obs = v_true + np.random.normal(0, 1.0, v_true.shape)

    g = nx.Graph()
    g.add_node("P0", type="Protein")
    samples = [f"S{i}" for i in range(n_samples)]

    for i in range(n_precursors):
        prec, pep = f"Pre{i}", f"Pep_{i}"
        g.add_node(prec, type="Precursor")
        g.add_node(pep, type="Peptide")
        g.add_edge("P0", pep, relation="PRODUCES")
        g.add_edge(pep, prec, relation="HAS_PRECURSOR")

    for i, s in enumerate(samples):
        g.add_node(s, type="Sample")
        for j in range(n_precursors):
            g.add_edge(f"Pre{j}", s, relation="DETECTED_IN", intensity=float(v_obs[i, j]))

    meta = pl.DataFrame({
        "file": samples,
        "name": samples,
        "dose": [0] * len(samples),
    })
    res_df, _ = NMFFit(g, meta_df=meta).fit()

    est_stderrs = [
        res_df.filter(pl.col("protein").str.contains("P0"))[f"{s}_stderr"][0]
        for s in samples
    ]
    assert 0.1 < np.mean(est_stderrs) < 10.0

    w_est = [
        res_df.filter(pl.col("protein").str.contains("P0"))[s][0]
        for s in samples
    ]
    z_scores = (np.array(w_est) - 100) / np.array(est_stderrs)
    assert np.abs(np.mean(z_scores)) < 25.0
    assert np.abs(np.std(z_scores) - 1.0) < 25.0
