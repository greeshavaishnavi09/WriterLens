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


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path

    ingestion_dir: Path

    inventory_file: Path

    image_inspection_file: Path

    xml_inspection_file: Path

    validation_report_path: Path

    writer_statistics_path: Path

    image_reference_errors_path: Path

    validation_summary_path: Path

    strict_mode: bool

    fail_on_missing_reference: bool

    fail_on_missing_writer_id: bool

    fail_on_duplicate_reference: bool

    check_writer_split_overlap: bool