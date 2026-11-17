import os
import sys

# make the repo root importable so `import screener` works from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
