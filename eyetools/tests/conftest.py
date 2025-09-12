import sys
from pathlib import Path

# Ensure root tools directory (containing classification) is on sys.path
TOOLS_ROOT = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))