"""Global pytest configuration and root fixtures for q1_response_predictor."""

import sys
from pathlib import Path

# Ensure SUBPROJECT_ROOT and PROJECT_ROOT are on sys.path so test discovery and
# modules can import src.* and scripts.* directly.
SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SUBPROJECT_ROOT.parent

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
