"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

# Keep API tests lightweight by skipping model load in app lifespan.
os.environ.setdefault("TEXTSUMMARIZER_SKIP_MODEL_LOAD", "1")

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
