"""API entrypoint.

Usage:
    pip install -r requirements.txt
    python app.py

The REST API has no authentication and exposes DELETE endpoints, so it should
only listen on the local loopback unless you explicitly opt in:

    HOST=127.0.0.1 python app.py   # local-only (recommended)
    HOST=0.0.0.0 python app.py     # network-exposed (needed inside Docker)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from loadguard.api import app  # noqa: E402

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=8000)
