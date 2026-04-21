#!/usr/bin/env python3
"""
download_patric.py

Download and parse PATRIC/BV-BRC AMR dataset for PREDICT-AMR proof-of-concept.

Usage:
    python download_patric.py --output data/ast_data.csv
    python download_patric.py --species "Escherichia coli" --output data/ast_filtered.csv
"""

import argparse
import pandas as pd
import urllib.request
import sys
import os
from pathlib import Path

# PATRIC AMR data URL (verified as of 2026)
PATRIC_AMR_URL = "ftp://ftp.bvbrc.org/RELEASE_NOTES/PATRIC_genomes_AMR.txt"

# Column names based on PATRIC documentation
COLUMN_NAMES = [
    "genome_id",
    "genome_name",
    "species",
    "genus",
    "antimicrobial",
    "resistance_phenotype",
    "mic_value",
    "mic_units",
    "source",
    "pubmed_id",
    "laboratory",
    "country",
    "year",
]


def download_patric_data(output_path: str) -> str:
    """
    Download PATRIC AMR data file from BV-BRC FTP server.

    Args:
        output_path: Path to save downloaded file

    Returns:
        Path to downloaded file
    """
    print(f"Downloading PATRIC AMR data from {PATRIC_AMR_URL}...")

    try:
        urllib.request.urlretrieve(PATRIC_AMR_URL, output_path)
        print(f"Download complete. File saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. The FTP server may be temporarily unavailable")
        print("3. Alternative: Download manually from https://www.bv-brc.org/")
        sys.exit(1)


def parse_patric_data(input_path: str, output_path: str, species_filter: str = None) -> pd.DataFrame:
    """
    Parse PATRIC AMR data file into structured DataFrame.

    Args:
        input_path: Path to downloaded PATRIC file
        output_path: Path to save parsed CSV
        species_filter: Optional species name to filter (e.g., "Escherichia coli")

    Returns:
        Parsed DataFrame
    """
    print(f"Parsing PATRIC data from {input_path}...")

    # Read tab-separated file
    try:
        df = pd.read_csv(input_path, sep="\t", header=None, names=COLUMN_NAMES, low_memory=False)
        print(f"Loaded {len(df):,} records")
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    # Filter by species if specified
    if species_filter:
        df = df[df["species"] == species_filter]
        print(f"Filtered to {species_filter}: {len(df):,} records")

    # Remove records with missing resistance phenotype
    df = df[df["resistance_phenotype"].notna()]
    print(f"Records with phenotype data: {len(df):,}")

    # Standardize phenotype to binary (R vs S)
    df["is_resistant"] = df["resistance_phenotype"].str.upper().apply(
        lambda x: 1 if x in ["R", "RESISTANT", "NS"] else 0 if x in ["S", "SUSCEPTIBLE"] else None
    )

    # Remove records where phenotype couldn't be mapped
    df = df[df["is_resistant"].notna()]
    print(f"Records with binary phenotype: {len(df):,}")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Parsed data saved to: {output_path}")

    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the dataset.

    Args:
        df: Parsed DataFrame
    """
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"\nTotal records: {len(df):,}")
    print(f"Total genomes: {df['genome_id'].nunique():,}")
    print(f"Total species: {df['species'].nunique()}")
    print(f"Total antimicrobials: {df['antimicrobial'].nunique()}")

    print("\nTop 10 species by record count:")
    species_counts = df["species"].value_counts().head(10)
    for species, count in species_counts.items():
        print(f"  {species}: {count:,}")

    print("\nTop 10 antimicrobials by record count:")
    drug_counts = df["antimicrobial"].value_counts().head(10)
    for drug, count in drug_counts.items():
        print(f"  {drug}: {count:,}")

    print("\nCountries represented:")
    countries = df["country"].dropna().value_counts().head(10)
    for country, count in countries.items():
        print(f"  {country}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Download and parse PATRIC AMR dataset")
    parser.add_argument("--output", type=str, default="data/ast_data.csv", help="Output CSV path")
    parser.add_argument("--species", type=str, default=None, help="Filter by species name")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    parser.add_argument("--no-download", action="store_true", help="Skip download (use existing file)")

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Download or use existing
    if not args.no_download:
        raw_path = args.output.replace(".csv", "_raw.txt")
        download_patric_data(raw_path)
        input_path = raw_path
    else:
        input_path = args.output.replace(".csv", "_raw.txt")
        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
            print("Run without --no-download first to download the data.")
            sys.exit(1)

    # Parse data
    df = parse_patric_data(input_path, args.output, species_filter=args.species)

    # Print summary if requested
    if args.summary:
        get_data_summary(df)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
