#!/usr/bin/env python3
"""
kmer_extractor.py

Extract k-mer presence/absence features from bacterial genome assemblies.

Usage:
    python kmer_extractor.py --input data/genomes/ --k 13 --step 1000 --output features/kmer_matrix.npz
    python kmer_extractor.py --input data/genomes/ --k 7 --max_kmers 500 --output features/kmer_matrix_small.npz
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import gzip
import hashlib


def read_fasta(filepath: str) -> dict:
    """
    Read FASTA file and return dictionary of sequence ID to sequence.

    Args:
        filepath: Path to FASTA file (may be .gz compressed)

    Returns:
        Dictionary {sequence_id: sequence_string}
    """
    sequences = {}
    current_id = None
    current_seq = []

    # Handle gzipped files
    open_func = gzip.open if filepath.endswith(".gz") else open

    with open_func(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # Take first word as ID
                current_seq = []
            else:
                current_seq.append(line.upper())

    if current_id:
        sequences[current_id] = "".join(current_seq)

    return sequences


def extract_kmers(sequence: str, k: int, step: int = 1) -> list:
    """
    Extract k-mers from a sequence with optional stepping.

    Args:
        sequence: DNA sequence string
        k: k-mer length
        step: Step size (use >1 for sampling)

    Returns:
        List of k-mers
    """
    kmers = []
    seq_len = len(sequence)

    for i in range(0, seq_len - k + 1, step):
        kmer = sequence[i : i + k]
        # Only include kmers with valid nucleotides (A, C, G, T)
        if all(base in "ACGT" for base in kmer):
            kmers.append(kmer)

    return kmers


def get_kmers_from_genomes(genome_dir: str, k: int = 13, step: int = 1000) -> dict:
    """
    Extract k-mers from all genome files in a directory.

    Args:
        genome_dir: Directory containing FASTA files
        k: k-mer length
        step: Step size for sampling

    Returns:
        Dictionary {genome_id: list_of_kmers}
    """
    genome_kmers = {}

    # Find all FASTA files
    fasta_files = []
    for ext in [".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz"]:
        fasta_files.extend(Path(genome_dir).glob(f"*{ext}"))

    if not fasta_files:
        print(f"Error: No FASTA files found in {genome_dir}")
        print("Supported extensions: .fa, .fasta, .fna, .fa.gz, .fasta.gz")
        sys.exit(1)

    print(f"Found {len(fasta_files)} genome files")
    print(f"Extracting {k}-mers with step={step}...")

    for filepath in tqdm(fasta_files, desc="Processing genomes"):
        sequences = read_fasta(str(filepath))

        for seq_id, sequence in sequences.items():
            kmers = extract_kmers(sequence, k, step)
            genome_kmers[seq_id] = kmers

    print(f"Extracted kmers from {len(genome_kmers)} genomes")
    return genome_kmers


def build_kmer_matrix(genome_kmers: dict, max_kmers: int = None) -> tuple:
    """
    Build binary presence/absence matrix of k-mers across genomes.

    Args:
        genome_kmers: Dictionary {genome_id: list_of_kmers}
        max_kmers: Maximum number of k-mers to include (for memory efficiency)

    Returns:
        X: Binary matrix (n_genomes x n_kmers)
        genome_ids: List of genome IDs
        kmer_list: List of k-mers (columns)
    """
    print("Building k-mer presence/absence matrix...")

    # Count k-mer frequencies across all genomes
    kmer_counts = Counter()
    for kmers in genome_kmers.values():
        kmer_counts.update(set(kmers))  # Use set for presence/absence (not multiplicity)

    # Select most frequent k-mers
    if max_kmers:
        top_kmers = [kmer for kmer, _ in kmer_counts.most_common(max_kmers)]
    else:
        top_kmers = list(kmer_counts.keys())

    print(f"Selected {len(top_kmers):,} k-mers")

    # Create mapping from kmer to column index
    kmer_to_idx = {kmer: i for i, kmer in enumerate(top_kmers)}

    # Build matrix
    n_genomes = len(genome_kmers)
    X = np.zeros((n_genomes, len(top_kmers)), dtype=np.int8)

    genome_ids = list(genome_kmers.keys())

    for i, genome_id in enumerate(genome_ids):
        kmers_set = set(genome_kmers[genome_id])
        for kmer in kmers_set:
            if kmer in kmer_to_idx:
                X[i, kmer_to_idx[kmer]] = 1

    print(f"Matrix shape: {X.shape}")
    print(f"Matrix density: {X.sum() / X.size:.4%}")

    return X, genome_ids, top_kmers


def save_matrix(X: np.ndarray, genome_ids: list, kmer_list: list, output_path: str) -> None:
    """
    Save k-mer matrix to compressed NPZ file.

    Args:
        X: Binary matrix
        genome_ids: List of genome IDs
        kmer_list: List of k-mers
        output_path: Output file path (.npz)
    """
    np.savez_compressed(
        output_path,
        X=X,
        genome_ids=np.array(genome_ids, dtype=object),
        kmers=np.array(kmer_list, dtype=object),
    )
    print(f"Matrix saved to: {output_path}")


def load_matrix(input_path: str) -> tuple:
    """
    Load saved k-mer matrix from NPZ file.

    Args:
        input_path: Path to .npz file

    Returns:
        X, genome_ids, kmer_list
    """
    data = np.load(input_path, allow_pickle=True)
    X = data["X"]
    genome_ids = data["genome_ids"].tolist()
    kmer_list = data["kmers"].tolist()
    return X, genome_ids, kmer_list


def main():
    parser = argparse.ArgumentParser(description="Extract k-mer features from bacterial genomes")
    parser.add_argument("--input", type=str, required=True, help="Directory containing genome FASTA files")
    parser.add_argument("--output", type=str, default="features/kmer_matrix.npz", help="Output NPZ file path")
    parser.add_argument("--k", type=int, default=13, help="k-mer length (default: 13)")
    parser.add_argument("--step", type=int, default=1000, help="Step size for sampling (default: 1000)")
    parser.add_argument("--max_kmers", type=int, default=1000, help="Maximum number of k-mers to include")
    parser.add_argument("--load", action="store_true", help="Load and display info about existing matrix")

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.load:
        # Load existing matrix and display info
        X, genome_ids, kmer_list = load_matrix(args.output)
        print(f"\nMatrix info:")
        print(f"  Shape: {X.shape}")
        print(f"  Density: {X.sum() / X.size:.4%}")
        print(f"  Genomes: {len(genome_ids)}")
        print(f"  k-mers: {len(kmer_list)}")
        print(f"\nFirst 5 genome IDs: {genome_ids[:5]}")
        print(f"First 5 k-mers: {kmer_list[:5]}")
        return

    # Extract k-mers from genomes
    genome_kmers = get_kmers_from_genomes(args.input, args.k, args.step)

    if not genome_kmers:
        print("Error: No k-mers extracted. Check your input files.")
        sys.exit(1)

    # Build presence/absence matrix
    X, genome_ids, kmer_list = build_kmer_matrix(genome_kmers, max_kmers=args.max_kmers)

    # Save matrix
    save_matrix(X, genome_ids, kmer_list, args.output)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
