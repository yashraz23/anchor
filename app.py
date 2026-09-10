"""Hugging Face Spaces entrypoint.

Spaces runs `app.py` at the repository root. The demo itself lives in the
package so it can be imported and tested like everything else.

Requires ANCHOR_PG_DSN and ANTHROPIC_API_KEY as Space secrets: a Space has no
Postgres of its own, so the corpus has to be reachable over the network.
"""

from anchor.demo.app import build

demo = build()

if __name__ == "__main__":
    demo.launch()
