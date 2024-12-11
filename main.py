#!/usr/bin/env python3
from src.pipeline import pipeline
from argparse import ArgumentParser
from sys import exit
import logging


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s, %(asctime)s -- %(message)s")
    logger = logging.getLogger("CPSC 545 main")
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)
    args = parser.parse_args()

    pipeline(arguments=args, logger=logger)


if __name__ == "__main__":
    exit(main())