from __future__ import annotations

import logging
import os
from enum import IntEnum, auto
from typing import Dict, Optional, Union

import polars as pl
import plotly.graph_objects as go
from networkx import Graph

from nomad.utils import db, graph_ops
from nomad.utils.digestor import InSilicoDigestor
from nomad.utils.fasta_parser import FastaParser
from nomad.utils.nmf import NMFFit
from nomad.utils.plotting import StructuralEvidencePlot
from nomad.utils.quantification_parser import QuantificationParser


logger = logging.getLogger(__name__)


class _State(IntEnum):
    """Internal state of the Nomad pipeline."""

    EMPTY = auto()
    LOADED = auto()
    FITTED = auto()


class Nomad:
    """NOMAD pipeline orchestrator for protein isoform deconvolution.

    Attributes:
        graph: The protein-peptide-precursor graph.
        metadata: Experimental design metadata.
        quant_df: Deconvolved isoform intensities.
        db_path: Path to the results database.
        num_workers: Number of parallel workers.
    """

    def __init__(
        self,
        metadata: Optional[Union[str, pl.DataFrame]] = None,
        num_workers: Optional[int] = None,
        db_path: str = "artifacts/nomad_results.sqlite",
        _preserve_db: bool = False,
    ) -> None:
        """Initializes the NOMAD instance.

        Args:
            metadata: Path to CSV or a polars DataFrame.
            num_workers: Number of workers for parallel processing.
            db_path: Path to the results database.
            _preserve_db: Internal flag — set True only by load_from_db to
                retain existing results. Normal construction always starts
                with a clean database.
        """
        self.graph = Graph()
        self.metadata = None
        self.quant_df = None
        self.db_path = db_path
        self._state = _State.EMPTY
        self.engine: Optional[NMFFit] = None
        self.fasta_path: Optional[str] = None

        if metadata is not None:
            self.set_metadata(metadata)

        self.num_workers = num_workers or os.cpu_count() or 4

        if not _preserve_db and os.path.exists(db_path):
            os.remove(db_path)
            logger.info("Cleared stale database at %s — starting fresh.", db_path)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        db.init_db(self.db_path)

    @classmethod
    def load_from_db(cls, db_path: str, metadata: Union[str, pl.DataFrame]) -> Nomad:
        """Loads a previous analysis state from the database.

        Args:
            db_path: Path to the SQLite database.
            metadata: Corresponding experimental metadata.

        Returns:
            A Nomad instance in the FITTED state.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        instance = cls(metadata=metadata, db_path=db_path, _preserve_db=True)
        instance.quant_df = db.load_intensities(db_path)
        instance._state = _State.FITTED
        return instance

    def set_metadata(self, metadata: Union[str, pl.DataFrame]) -> None:
        """Sets the experimental metadata.

        Args:
            metadata: Path to CSV or a polars DataFrame.
        """
        self.metadata = (
            pl.read_csv(metadata) if isinstance(metadata, str) else metadata
        )

    def load_fasta(self, fasta_file: str) -> None:
        """Parses a FASTA file into the graph.

        Args:
            fasta_file: Path to the protein database.
        """
        self.fasta_path = fasta_file
        FastaParser(self.graph, fasta_file).parse()

    def digest(self, enzyme: str = "trypsin/p", **kwargs) -> None:
        """Performs in-silico digestion of proteins in the graph.

        Args:
            enzyme: Protease to use for digestion.
            **kwargs: Additional parameters for the digestor.
        """
        InSilicoDigestor(
            self.graph, enzyme, num_workers=self.num_workers, **kwargs
        ).digest()

    def load_quantification(
        self, quant_file: str, skip_rt_clustering: bool = False
    ) -> None:
        """Parses precursor quantification and joins with metadata.

        Args:
            quant_file: Path to the combined_ion.tsv file.
            skip_rt_clustering: Whether to skip RT-based clustering.
        """
        QuantificationParser(
            self.graph,
            quant_file,
            metadata=self.metadata,
            skip_rt_clustering=skip_rt_clustering,
        ).parse()
        self.graph = graph_ops.prune_graph(self.graph)
        self._state = _State.LOADED

    def quantify(
        self, fasta: str, ev: str, enz: str = "trypsin/p", **kwargs
    ) -> Dict[str, float]:
        """Runs the full deconvolution pipeline from inputs.

        Args:
            fasta: Path to FASTA.
            ev: Path to FragPipe quantification.
            enz: Enzyme for digestion.
            **kwargs: Fit parameters.

        Returns:
            Dictionary of cross-validation metrics.
        """
        self.load_fasta(fasta)
        self.digest(enz)
        self.load_quantification(ev)
        return self.fit(**kwargs)

    def fit(
        self,
        lambda_reg: float = 1e-3,
        gamma_dr: float = 10.0,
        beta_rep: float = 0.1,
        **kw,
    ) -> Dict:
        """Fits the NMF model to the graph signal.

        Args:
            lambda_reg: Regularization strength.
            gamma_dr: Dose-response manifold strength.
            beta_rep: Replicate manifold strength.
            **kw: Solver parameters.

        Returns:
            Dictionary of CV metrics.
        """
        if self._state < _State.LOADED:
            raise RuntimeError("Call load_quantification() first.")

        self.engine = NMFFit(
            self.graph,
            meta_df=self.metadata,
            num_workers=self.num_workers,
            lambda_reg=lambda_reg,
            gamma_dr=gamma_dr,
            beta_rep=beta_rep,
        )
        self.quant_df, res = self.engine.fit(**kw)
        db.save_intensities(self.db_path, self.quant_df)
        db.save_emissions(self.db_path, self.engine.emissions_df)
        if self.engine.cv_data:
            db.save_diagnostic_loo(self.db_path, self.engine.cv_data)
        self._state = _State.FITTED
        return res

    def structural_evidence(
        self, protein: str, bottom_right: str = "reconstruction"
    ) -> go.Figure:
        """Visualizes the structural evidence matrix.

        Args:
            protein: UniProt ID.
            bottom_right: View mode for bottom-right panel.

        Returns:
            Plotly Figure object.
        """
        if self._state < _State.FITTED:
            raise RuntimeError("Requires FITTED state.")
        return StructuralEvidencePlot.plot(
            self.db_path, protein, self.metadata, bottom_right
        )

    def plot_performance(self, output_dir: Optional[str] = "artifacts/plots") -> go.Figure:
        """Generates NMF deconvolution diagnostic plots (LOO and Replicates).

        Args:
            output_dir: Optional directory to save the diagnostic HTML.

        Returns:
            Plotly Figure object.
        """
        if self.engine is None or not self.engine.cv_data:
            raise RuntimeError("Requires a completed fit() to generate diagnostics.")

        from nomad.utils.plotting import DiagnosticsPlot
        return DiagnosticsPlot.plot_nmf_performance(
            self.engine.cv_data, self.engine.rep_data, self.engine.rs_data, output_dir=output_dir
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the NOMAD deconvolution pipeline.")
    parser.add_argument("--fasta", "-f", required=True, help="Path to FASTA database")
    parser.add_argument("--evidence", "-e", required=True, help="Path to combined_ion.tsv")
    parser.add_argument("--metadata", "-m", default="experiment/data/dose_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--out", "-o", default="artifacts/nomad_quant.tsv", help="Path to output quantification TSV")
    parser.add_argument("--db", "-d", default="artifacts/nomad_results.sqlite", help="Path to SQLite database")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    if os.path.exists(args.db) and os.path.getsize(args.db) > 100000:
        print(f"[*] Found existing completed database at {args.db}. Loading cached results...")
        nm = Nomad.load_from_db(args.db, args.metadata)
    else:
        print("[*] Initializing NOMAD...")
        nm = Nomad(metadata=args.metadata, num_workers=args.workers, db_path=args.db)
        
        print("[*] Loading FASTA and digesting...")
        nm.load_fasta(args.fasta)
        nm.digest(enzyme="trypsin/p")
        
        print("[*] Loading precursors and NMF fitting...")
        nm.load_quantification(args.evidence)
        nm.fit()
    
    print("[*] Writing NOMAD quantification TSV...")
    fixed_cols = ["protein", "gene_symbol", "entry_name", "description", "n_proteins"]
    other_cols = [c for c in nm.quant_df.columns if c not in fixed_cols and c != "structural_specificity"]
    out_df = nm.quant_df.select(fixed_cols + other_cols)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.write_csv(args.out, separator="\t")
    print(f"[+] Output written to: {args.out}")


if __name__ == "__main__":
    main()


