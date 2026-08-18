"""Guard against README/docs drift.

The README Mermaid diagram is generated from ``docs/architecture.mmd``; this
test fails if the two get out of sync, so an exaggerated or stale claim can't
slip back into the README unnoticed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestDocsSync(unittest.TestCase):
    def test_readme_diagram_matches_mmd(self) -> None:
        mmd = (ROOT / "docs" / "architecture.mmd").read_text(encoding="utf-8")
        # Strip leading %% comment lines from the .mmd source.
        mmd_body = "\n".join(
            line for line in mmd.splitlines() if not line.lstrip().startswith("%%")
        ).lstrip("\n")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index("```mermaid\n") + len("```mermaid\n")
        end = readme.index("\n```", start)
        readme_body = readme[start:end].rstrip("\n")

        self.assertEqual(
            mmd_body,
            readme_body,
            "README Mermaid diagram is out of sync with docs/architecture.mmd",
        )


if __name__ == "__main__":
    unittest.main()
