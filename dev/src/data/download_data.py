#!/usr/bin/env python3
"""
Data Download Script for Historia Scribe

This script handles downloading and extracting datasets for training and evaluation.
Supports multiple historical handwriting datasets as specified in the roadmap.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def download_iam_dataset(output_dir: Path) -> None:
    """
    Download and extract the IAM Handwriting Database.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print(f"Downloading IAM dataset to {output_dir}")
    # TODO: Implement IAM dataset download
    # The IAM database is available on Hugging Face Datasets as 'Teklia/IAM-line'
    # or can be downloaded from official sources
    print("IAM dataset download not yet implemented")


def download_bentham_dataset(output_dir: Path) -> None:
    """
    Download and extract the Bentham Collection dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print(f"Downloading Bentham dataset to {output_dir}")
    # TODO: Implement Bentham dataset download
    # Available through UCL Digital Collections and Transcribe Bentham project
    print("Bentham dataset download not yet implemented")


def download_read_icfhr_dataset(output_dir: Path) -> None:
    """
    Download and extract the READ-ICFHR 2016 dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print(f"Downloading READ-ICFHR dataset to {output_dir}")
    # TODO: Implement READ-ICFHR dataset download
    # Available on Zenodo
    print("READ-ICFHR dataset download not yet implemented")


def main() -> None:
    """Main function to handle dataset downloads."""
    parser = argparse.ArgumentParser(
        description="Download datasets for Historia Scribe"
    )
    parser.add_argument(
        "dataset",
        choices=["IAM", "Bentham", "READ-ICFHR"],
        help="Name of the dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the specified dataset
    if args.dataset == "IAM":
        download_iam_dataset(args.output_dir)
    elif args.dataset == "Bentham":
        download_bentham_dataset(args.output_dir)
    elif args.dataset == "READ-ICFHR":
        download_read_icfhr_dataset(args.output_dir)
    
    print(f"Dataset {args.dataset} download completed")


if __name__ == "__main__":
    main()
