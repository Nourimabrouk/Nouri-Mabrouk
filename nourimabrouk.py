"""Top-level wrapper to expose agents.nourimabrouk as a flat module.

This preserves imports like `from nourimabrouk import NouriMabrouk` that
exist in agent modules.
"""

from agents.nourimabrouk import *  # noqa: F401,F403

