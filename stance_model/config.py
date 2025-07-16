from pathlib import Path

LABELS = ["Pro-Palestine", "irrelevant", "Pro-Israel", "Neutral"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}
OUTPUT_DIR = Path.cwd() / "stance_model"
EXPORT_DIR = Path.cwd() / "exported_stance_model"
