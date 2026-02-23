"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
