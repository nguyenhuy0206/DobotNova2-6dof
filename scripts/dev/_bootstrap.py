"""Make ``src/`` importable for the scripts in scripts/dev/ (as scripts/_bootstrap.py does one level up)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
# always first: an editable install may already list src/ at the END of sys.path, and then a
# script in the script's own folder with a package's name (e.g. scripts/student.py) would win
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))
