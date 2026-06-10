"""Module for parsing FragPipe combined_ion.tsv files and updating the Graph."""

import os
from typing import Optional

import networkx as nx
import polars as pl


class QuantificationParser:
    """Parses FragPipe combined_ion.tsv files to update the graph with observations.

    Attributes:
        g: The networkx graph to update.
        file_path: Path to the quantification file.
        metadata: Optional metadata for sample mapping.
    """

    def __init__(self, g: nx.Graph, file_path: str, metadata: Optional[pl.DataFrame] = None,
                 **kwargs):
        """Initializes the FragPipe combined_ion.tsv parser.

        Args:
            g: Graph to update.
            file_path: Path to combined_ion.tsv file.
            metadata: Optional metadata containing 'sample' definitions.
        """
        self.g = g
        self.file_path = file_path
        self.metadata = metadata
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def parse(self) -> None:
        """Parses FragPipe combined_ion.tsv and updates the graph."""
        df = pl.read_csv(self.file_path, separator="\t", infer_schema_length=10000)
        if "Peptide Sequence" not in df.columns or "Charge" not in df.columns:
            raise ValueError("Only FragPipe combined_ion.tsv format is supported.")

        # Extract sample intensity columns
        intensity_cols = [c for c in df.columns if c.endswith(" Intensity")]
        if not intensity_cols:
            print(" [!] No intensity columns found in combined_ion.tsv.")
            return

        # 1. Unpivot intensities to long format
        df_long = df.select([
            pl.col("Peptide Sequence").alias("peptide_sequence"),
            pl.col("Charge").alias("charge"),
            *intensity_cols
        ]).unpivot(
            index=["peptide_sequence", "charge"],
            variable_name="sample_raw",
            value_name="intensity"
        ).filter(
            pl.col("intensity").is_finite() & (pl.col("intensity") > 0)
        )

        # Clean sample names: strip ' Intensity' and trailing FragPipe suffixes (e.g., '_1')
        df_long = df_long.with_columns(
            pl.col("sample_raw").str.replace(" Intensity", "").str.replace(r"_\d+$", "").alias("sample_id")
        ).drop("sample_raw")

        # Optional metadata validation
        if self.metadata is not None and "sample" in self.metadata.columns:
            valid_samples = self.metadata["sample"].unique().to_list()
            # Keep only samples defined in metadata if any match
            if df_long.filter(pl.col("sample_id").is_in(valid_samples)).height > 0:
                df_long = df_long.filter(pl.col("sample_id").is_in(valid_samples))

        # 2. Replicate-aware normalization
        df_long = self._normalize_intensities(df_long)

        # 3. Define precursor purely by sequence and charge, aggregate raw sums
        df_agg = self._aggregate_intensities(df_long)
        if df_agg.is_empty():
            print(" [!] No quantification data found after aggregation.")
            return

        print(f" [*] Found {df_agg.height} observations.")
        df_filtered = self._filter_df(df_agg)
        if df_filtered.is_empty():
            print(" [!] No observations matching digestion and detected in >=3 samples.")
            return

        print(f" [*] Filtered down to {df_filtered.height} observations matching digestion (retaining only those seen in >=3 samples).")
        self._update_graph(df_filtered)
        print(" [*] Quantification parsing complete.")

    def _normalize_intensities(self, df: pl.DataFrame) -> pl.DataFrame:
        """Performs median intensity normalization to align sample TICs."""
        # 1. Per-sample medians
        sample_medians = df.group_by("sample_id").agg(
            pl.col("intensity").median().alias("sample_median")
        )
        
        # 2. Global target (median of all sample medians)
        global_target = sample_medians["sample_median"].median()
        if global_target is None or global_target == 0:
            return df
            
        print(f" [*] Normalizing intensities to global target: {global_target:.2e}")
        
        # 3. Scale every sample to the global target
        df = df.join(sample_medians, on="sample_id")
        df = df.with_columns(
            (pl.col("intensity") * (global_target / pl.col("sample_median"))).alias("intensity")
        )
        return df.drop("sample_median")

    def _aggregate_intensities(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates raw intensities by precursor (sequence + charge)."""
        df = df.with_columns(
            pl.col("peptide_sequence").str.replace_all("_", "").str.replace_all(r"\(.*?\)", "")
        )
        df = df.with_columns(
            (pl.col("peptide_sequence") + "_" + pl.col("charge").cast(pl.Utf8)).alias("precursor_id")
        )
        return df.group_by(["peptide_sequence", "precursor_id", "sample_id"]).agg([
            pl.col("intensity").sum().alias("intensity"),
        ])

    def _filter_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters observations matching existing graph Peptides and requires >=3 samples."""
        peptides_in_graph = [n for n, d in self.g.nodes(data=True) if d.get("type") == "Peptide"]
        df = df.filter(pl.col("peptide_sequence").is_in(peptides_in_graph))

        precursor_counts = df.group_by("precursor_id").len()
        valid_precursors = precursor_counts.filter(pl.col("len") >= 3)["precursor_id"].to_list()
        return df.filter(pl.col("precursor_id").is_in(valid_precursors))

    def _update_graph(self, df: pl.DataFrame) -> None:
        """Adds Sample, Precursor nodes and edges to the graph."""
        samples = df["sample_id"].unique().to_list()
        self.g.add_nodes_from((s, {"type": "Sample"}) for s in samples)

        unique_precursors = df.unique("precursor_id")
        prec_ids = unique_precursors["precursor_id"].to_list()
        pep_seqs = unique_precursors["peptide_sequence"].to_list()

        self.g.add_nodes_from((p, {"type": "Precursor"}) for p in prec_ids)
        self.g.add_edges_from((seq, p, {"relation": "HAS_PRECURSOR"}) for seq, p in zip(pep_seqs, prec_ids))

        prec_col = df["precursor_id"].to_list()
        samp_col = df["sample_id"].to_list()
        int_col = df["intensity"].to_list()

        self.g.add_edges_from(
            (p, s, {"relation": "DETECTED_IN", "intensity": float(val)})
            for p, s, val in zip(prec_col, samp_col, int_col)
        )