
from Handwriting_Verification_Using_Siamese_Neural_Networks.config.configuration import ConfigurationManager
from Handwriting_Verification_Using_Siamese_Neural_Networks.logging import logger
from Handwriting_Verification_Using_Siamese_Neural_Networks.components.data_ingestion import DataIngestion


# Pipeline

class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        
        try:
            print("\n--- STARTING DATA INGESTION PIPELINE ---")

            config = ConfigurationManager()

            data_ingestion_config = (config.get_data_ingestion_config())

            data_ingestion = DataIngestion(config=data_ingestion_config)

            results = data_ingestion.initiate_data_ingestion()

            print("\n--- DATA INGESTION PIPELINE COMPLETED ---")

            return results

        except Exception:
            logger.exception("Data Ingestion Pipeline failed.")
            raise