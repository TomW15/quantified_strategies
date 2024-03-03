from pathlib import Path


def get_path(path: str) -> Path:
    return Path(path).parent
