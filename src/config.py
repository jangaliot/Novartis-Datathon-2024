from pathlib import Path

import os

from dotenv import load_dotenv

# Paths
REPO_DIR = Path(os.getenv('REPO_PATH'))

# Data paths
DATA_DIR = REPO_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'