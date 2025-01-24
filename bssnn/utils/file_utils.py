from pathlib import Path
from datetime import datetime

def create_timestamped_dir(base_dir: str) -> Path:
    """Create a timestamped directory within the base directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = Path(base_dir) / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir