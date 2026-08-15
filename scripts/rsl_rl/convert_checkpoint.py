"""Convert an RSL-RL 2/3 checkpoint to the RSL-RL 5 layout."""

from __future__ import annotations

import argparse

from whole_body_tracking.utils.checkpoint import convert_checkpoint_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Legacy checkpoint path.")
    parser.add_argument("destination", help="Converted checkpoint path.")
    args = parser.parse_args()
    output = convert_checkpoint_file(args.source, args.destination)
    print(f"[INFO] Converted checkpoint written to: {output}")


if __name__ == "__main__":
    main()
