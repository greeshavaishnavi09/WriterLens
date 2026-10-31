from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:

    root_dir: Path
    dataset_name: str
    data_source: Path
    dataset_dir: Path
    train_dir: Path
    test_dir: Path
    image_types: tuple
    annotation_dir: str