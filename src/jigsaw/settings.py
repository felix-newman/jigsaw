from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
