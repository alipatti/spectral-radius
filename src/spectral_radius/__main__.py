"""Replicate the paper's data, figures, and appendix; with no command, all three."""

import argparse

from spectral_radius import appendix, examples, spectral
from spectral_radius.surveys import build


def figures() -> None:
    spectral.main()
    examples.main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m spectral_radius")
    commands = parser.add_subparsers(dest="command")
    data = commands.add_parser("data", help="build data/clean/*.parquet")
    data.add_argument(
        "--refresh",
        action="store_true",
        help="redownload the raw data",
    )
    commands.add_parser("figures", help="draw every figure")
    commands.add_parser("appendix", help="write the questions appendix")
    args = parser.parse_args()

    match args.command:
        case "data":
            build(refresh=args.refresh)
        case "figures":
            figures()
        case "appendix":
            appendix.main()
        case None:
            build()
            figures()
            appendix.main()


if __name__ == "__main__":
    main()
