#!/usr/bin/env python3
from src.pipeline import pipeline
from argparse import ArgumentParser
from sys import exit
from pathlib import Path
import logging


def main():
    """Main function for the pipeline of the CPSC 545 final project.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s, %(asctime)s -- %(message)s")
    logger = logging.getLogger("CPSC 545 main")
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Output directory", required=True)
    args = parser.parse_args()

    pipeline(output_dir=Path(args.output_dir), logger=logger)


if __name__ == "__main__":
    exit(main())