"""Module for in-silico digestion of protein sequences."""

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import networkx as nx
import polars as pl
from tqdm import tqdm

from nomad.utils.constants import ENZYME_REGEX


def _parallel_digest_worker(batch: List[Tuple[str, str]], cleave_regex: str,
                            missed_cleavages: int, min_pep_len: int,
                            max_pep_len: int, allow_met_excision: bool = True) -> List[Dict[str, str]]:
    """Parallel digestion worker for processing a batch of proteins.

    Args:
        batch: List of tuples containing (protein_id, sequence_str).
        cleave_regex: Regular expression for enzyme cleavage.
        missed_cleavages: Maximum number of missed cleavages allowed.
        min_pep_len: Minimum peptide length to retain.
        max_pep_len: Maximum peptide length to retain.
        allow_met_excision: Whether to also digest the sequence without N-terminal methionine.

    Returns:
        A list of dictionaries, each containing 'protein_id' and 'peptide_sequence'.
    """
    peptides = []
    for protein_id, sequence_str in batch:
        sequence = str(sequence_str).upper().replace("*", "").replace(
            "-", "").strip()
        sequence = re.sub(r"[\[\{].*?[\]\}]", "", sequence)
        if not sequence:
            continue

        sequences_to_digest = [sequence]
        if allow_met_excision and sequence.startswith("M") and len(sequence) > 1 and sequence[1] in {"G", "A", "S", "T", "C", "P", "V"}:
            sequences_to_digest.append(sequence[1:])

        for seq in sequences_to_digest:
            fragments = [p for p in re.split(cleave_regex, seq) if p]
            for mc in range(missed_cleavages + 1):
                for i in range(len(fragments) - mc):
                    joined_pep = "".join(fragments[i:i + mc + 1])
                    if min_pep_len <= len(joined_pep) <= max_pep_len:
                        peptides.append({
                            "protein_id": protein_id,
                            "peptide_sequence": joined_pep
                        })
    return peptides


class InSilicoDigestor:
    """Digests protein sequences into peptides using specified enzyme parameters.

    Attributes:
        g: The networkx graph containing protein nodes.
        missed_cleavages: Allowed missed cleavages.
        cleave_regex: Compiled regex for cleavage.
        num_workers: Number of parallel workers to use.
        min_pep_len: Minimum peptide length.
        max_pep_len: Maximum peptide length.
    """

    def __init__(self,
                 g: nx.Graph,
                 enzyme: str = "trypsin",
                 missed_cleavages: int = 2,
                 num_workers: int = 4,
                 min_pep_len: int = 6,
                 max_pep_len: int = 50,
                 allow_met_excision: bool = True):
        """Initializes the digestor with graph and enzyme parameters.

        Args:
            g: Graph with protein sequences.
            enzyme: Enzyme name (e.g., 'trypsin').
            missed_cleavages: Allowed missed cleavages.
            num_workers: Number of parallel workers for digestion.
            min_pep_len: Minimum peptide length.
            max_pep_len: Maximum peptide length.
            allow_met_excision: Whether to include Met-excised N-terminal peptides.
        """
        if enzyme.lower() not in ENZYME_REGEX:
            raise ValueError(f"Enzyme '{enzyme}' not supported.")
        self.g = g
        self.missed_cleavages = missed_cleavages
        self.cleave_regex = ENZYME_REGEX.get(enzyme.lower())
        self.num_workers = num_workers
        self.min_pep_len = min_pep_len
        self.max_pep_len = max_pep_len
        self.allow_met_excision = allow_met_excision

    def digest(self) -> None:
        """Executes in-silico digestion and updates the graph."""
        protein_list, num_proteins = self._extract_proteins()
        if not protein_list:
            print(" [!] No proteins with sequences found in graph.")
            return

        print(f" [*] Digesting {num_proteins} proteins using "
              f"{self.num_workers} workers...")

        batch_size = max(1, num_proteins // (max(1, self.num_workers) * 2))
        batches = [
            protein_list[i:i + batch_size]
            for i in range(0, num_proteins, batch_size)
        ]

        peptides = self._run_parallel_digestion(batches)
        if not peptides:
            print(" [!] No valid peptides generated.")
            return

        peptides_df = pl.DataFrame(peptides)

        unique_count = peptides_df.select(
            pl.col("peptide_sequence").n_unique()).item()
        print(f" [*] Adding {unique_count} peptides and "
              f"{peptides_df.height} edges...")

        self._update_graph(peptides_df)
        print(" [*] Digestion complete.")

    def _extract_proteins(self) -> Tuple[List[Tuple[str, str]], int]:
        """Retrieves protein nodes and sequences from the graph.

        Returns:
            A tuple containing a list of (protein_id, sequence) and the total count.
        """
        proteins = [(node, data['sequence'])
                    for node, data in self.g.nodes(data=True)
                    if data.get('type') == 'Protein' and 'sequence' in data]
        return proteins, len(proteins)

    def _run_parallel_digestion(
            self, batches: List[List[Tuple[str, str]]]) -> List[Dict[str, str]]:
        """Runs digestion in parallel workers (or serial if num_workers=1)."""
        all_peptides = []

        if self.num_workers <= 1:
            for batch in tqdm(batches, desc=" [*] Digesting batches"):
                all_peptides.extend(_parallel_digest_worker(
                    batch, self.cleave_regex, self.missed_cleavages,
                    self.min_pep_len, self.max_pep_len, self.allow_met_excision
                ))
            return all_peptides

        # Use spawn context to avoid fork safety issues with threads
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_parallel_digest_worker, batch,
                                self.cleave_regex, self.missed_cleavages,
                                self.min_pep_len, self.max_pep_len, self.allow_met_excision)
                for batch in batches
            ]
            for future in tqdm(futures, desc=" [*] Digesting batches"):
                all_peptides.extend(future.result())
        return all_peptides

    def _update_graph(self, peptides_df: pl.DataFrame) -> None:
        """Adds peptide nodes and edges to the graph.

        Args:
            peptides_df: Polars DataFrame containing peptide data.
        """
        # Node generation: Use unique() on column directly to avoid expensive row-dict iteration
        unique_peps = peptides_df["peptide_sequence"].unique()
        self.g.add_nodes_from((pep, {"type": "Peptide"}) for pep in unique_peps)

        # Edge generation: Use zip() on columns for massive performance gain over iter_rows
        # All edges share the same attribute dictionary to save memory (references are stored)
        attr = {"relation": "PRODUCES"}
        self.g.add_edges_from(zip(
            peptides_df["protein_id"],
            peptides_df["peptide_sequence"],
            [attr] * len(peptides_df)
        ))