import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.smoke_test import run_smoke_test

if __name__ == "__main__":
    run_smoke_test()
