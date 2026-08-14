"""Allows `python -m carmine ...` as an alias for the `carmine` console script."""

import sys

from carmine.cli import main

if __name__ == "__main__":
    sys.exit(main())
