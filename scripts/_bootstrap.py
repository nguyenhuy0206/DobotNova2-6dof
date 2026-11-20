"""Make ``src/`` importable when running scripts without ``pip install -e .``."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
# always first: an editable install may already list src/ at the END of sys.path, and then a
# script in the script's own folder with a package's name (e.g. scripts/student.py) would win
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))
