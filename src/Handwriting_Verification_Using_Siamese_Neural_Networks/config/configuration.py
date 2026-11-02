from Handwriting_Verification_Using_Siamese_Neural_Networks.constant import CONFIG_FILE_PATH,PARAMS_FILE_PATH,PROJECT_ROOT
from Handwriting_Verification_Using_Siamese_Neural_Networks.constant import *
from Handwriting_Verification_Using_Siamese_Neural_Networks.utils.common import read_yaml,create_directories
from Handwriting_Verification_Using_Siamese_Neural_Networks.entity.config_entity import DataIngestionConfig


# CONFIGURATION MANAGER

class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        # Read YAML configuration.
        self.config = read_yaml(config_filepath)

        # Read experiment parameters.
        self.params = read_yaml(params_filepath)

        # Create artifacts root directory.
        create_directories([PROJECT_ROOT / self.config.artifacts_root])


    # DATA INGESTION CONFIGURATION
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        # Resolve dataset source path.
        data_source = PROJECT_ROOT / config.data_source

        if not data_source.exists():
            raise FileNotFoundError(f"Dataset source not found: {data_source}")

        # Locate the extracted CVL dataset directory.
        dataset_candidates = [
            path.parent
            for path in data_source.rglob("trainset")
            if path.is_dir()
            and (path.parent / "testset").is_dir()
        ]

        if len(dataset_candidates) != 1:
            raise ValueError(
                f"Expected exactly one CVL dataset folder, "
                f"but found: {dataset_candidates}"
            )

        dataset_dir = dataset_candidates[0]

        train_dir = dataset_dir / config.train_dir
        test_dir = dataset_dir / config.test_dir

        # Create ingestion artifact directory.
        root_dir = PROJECT_ROOT / config.root_dir

        create_directories([root_dir])

        # Create configuration entity.
        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            dataset_name=config.dataset_name,
            data_source=data_source,
            dataset_dir=dataset_dir,
            train_dir=train_dir,
            test_dir=test_dir,
            image_types=tuple(config.image_types),
            annotation_dir=config.annotation_dir,
        )

        return data_ingestion_config
