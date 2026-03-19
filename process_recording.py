"""Entry point for the cello sample extraction and classification tool.

Run with::

    python process_recording.py INPUT_FILE OUTPUT_DIR [options]

For full help::

    python process_recording.py --help
"""

import sys

from cello_sampler.cli import main

if __name__ == "__main__":
    sys.exit(main())
