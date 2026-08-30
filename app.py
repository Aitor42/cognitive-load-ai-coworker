"""API entrypoint.

Usage:
    pip install -r requirements.txt
    python app.py

The REST API binds to the local loopback by default and supports optional API-key
authentication. Because it exposes DELETE endpoints, use authentication and disable
destructive operations when network-exposed:

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
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "8000"))
        if not (1 <= port <= 65535):
            port = 8000
    except ValueError:
        port = 8000
    uvicorn.run(app, host=host, port=port)
