import sys
from pathlib import Path

# Ensure the project root (the directory containing the `spark` package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
