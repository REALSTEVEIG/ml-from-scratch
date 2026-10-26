"""Allow ``python -m mlscratch``."""

import sys

from mlscratch.cli import main

if __name__ == "__main__":
    sys.exit(main())
