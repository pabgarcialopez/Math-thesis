import json
from pathlib import Path
from typing import List

def load_json(path: Path) -> object:
    """Load and return the contents of a JSON file."""
    with path.open() as f:
        return json.load(f)

def list_dirs(path: Path) -> List[Path]:
    """Return a sorted list of subdirectories under `path`."""
    return sorted(p for p in path.iterdir() if p.is_dir())