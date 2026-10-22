"""Load scientific regression tests from the retained runtime only.

The historical package names are part of serialized models and receipts. Tests
must exercise the same isolated sources as a real run, never an older checkout
or an unrelated installation with a matching import name.
"""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nyx_runtime.engine import activate_runtime


activate_runtime(PROJECT_ROOT / "production" / "runtime")
