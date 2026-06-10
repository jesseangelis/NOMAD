"""Module for parsing FASTA files and updating the Graph."""

import gzip
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import polars as pl
from tqdm import tqdm


class FastaParser:
    """Parses FASTA files to populate Protein nodes."""

    def __init__(self, g: nx.Graph, file_path: str):
        """Initializes the FASTA parser.

        Args:
            g: Graph to update.
            file_path: Path to FASTA file.

        Raises:
            FileNotFoundError: If the FASTA file does not exist.
        """
        self.g = g
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.file_path}")

    def _extract_id(self, header_line: str) -> str:
        """Extracts ID from FASTA header.

        Args:
            header_line: The raw header line from the FASTA file.

        Returns:
            The extracted protein ID.
        """
        line = header_line[1:].strip()
        if not line:
            return ""

        parts = line.split()
        raw_id = parts[0] if parts else ""

        if raw_id.count("|") >= 2:
            tokens = raw_id.split("|")
            if tokens[0] in ["sp", "tr"]:
                return tokens[1]

        return raw_id

    def parse(self) -> None:
        """Parses FASTA and updates the graph."""
        ids, sequences = self._parse_fasta_file()
        if not ids:
            print(" [!] No sequences found.")
            return

        new_data = self._create_protein_dataframe(ids, sequences)
        self._append_to_graph(new_data)
        
        # Report disambiguation
        n_disambig = new_data.filter(pl.col("id").str.contains(r"\$")).height
        if n_disambig > 0:
             print(f" [*] Found {n_disambig} isoforms with identical names but different sequences (annotated with $).")
             
        print(" [*] Parsing FASTA file complete.")

    def _parse_fasta_file(self) -> Tuple[List[str], List[str]]:
        """Reads FASTA file and extracts headers and sequences.

        Returns:
            A tuple containing a list of IDs and a list of sequences.
        """
        open_func = gzip.open if self.file_path.suffix == ".gz" else open
        ids = []
        sequences = []

        DECOY_PREFIXES = ["REV__", "CON__"]

        with open_func(self.file_path, "rt") as handle:
            header = None
            current_sequence = []

            for line in tqdm(handle, desc=" [*] Parsing FASTA file"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        # Only add if not a decoy
                        if not any(prefix in header for prefix in DECOY_PREFIXES):
                            ids.append(header)
                            sequences.append("".join(current_sequence).upper())
                    header = line
                    current_sequence = []
                else:
                    current_sequence.append(line)

            if header is not None:
                if not any(prefix in header for prefix in DECOY_PREFIXES):
                    ids.append(header)
                    sequences.append("".join(current_sequence).upper())

        return ids, sequences

    def _create_protein_dataframe(self, ids: List[str],
                                  sequences: List[str]) -> pl.DataFrame:
        """Creates a Polars DataFrame with disambiguated IDs if sequences differ.
        
        If a base parsed ID (e.g. P12345) is encountered multiple times:
        - If the sequence is identical, it is treated as a duplicate and ignored.
        - If the sequence is different, a $ suffix is added to disambiguate the ID.
        """
        seen_base_ids = {}  # base_id -> sequence
        id_counters = {}    # base_id -> current_count
        
        final_ids = []
        final_sequences = []
        
        for header, seq in zip(ids, sequences):
            base_id = self._extract_id(header)
            
            if base_id not in seen_base_ids:
                seen_base_ids[base_id] = seq
                final_ids.append(header)
                final_sequences.append(seq)
                id_counters[base_id] = 1
            else:
                if seen_base_ids[base_id] == seq:
                    # Identical sequence for same base ID: ignore duplicate
                    continue
                else:
                    # Conflict: same base ID, different sequence
                    id_counters[base_id] += 1
                    disambiguated_id = f"{base_id}${id_counters[base_id]}"
                    
                    # Inject disambiguated ID back into the original header string
                    if header.startswith(">sp|") or header.startswith(">tr|"):
                        parts = header.split("|")
                        parts[1] = disambiguated_id
                        new_header = "|".join(parts)
                    else:
                        parts = header.split(" ", 1)
                        parts[0] = f">{disambiguated_id}"
                        new_header = " ".join(parts)
                        
                    final_ids.append(new_header)
                    final_sequences.append(seq)
        
        return pl.DataFrame({
            "id": final_ids,
            "sequence": final_sequences
        })

    def _append_to_graph(self, new_data: pl.DataFrame) -> None:
        """Adds new protein nodes to the graph.

        Args:
            new_data: Polars DataFrame containing protein IDs and sequences.
        """
        node_generator = (
            (row["id"], {
                "type": "Protein",
                "sequence": row["sequence"]
            }) for row in new_data.iter_rows(named=True))
        self.g.add_nodes_from(node_generator)