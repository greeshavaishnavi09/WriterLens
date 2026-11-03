
from Handwriting_Verification_Using_Siamese_Neural_Networks.config.configuration import ConfigurationManager


from Handwriting_Verification_Using_Siamese_Neural_Networks.logging import logger

from Handwriting_Verification_Using_Siamese_Neural_Networks.components.data_ingestion import DataIngestion



class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        pipeline_start_message = (
            "STARTING DATA INGESTION PIPELINE"
        )

        try:

            print(f"\n--- {pipeline_start_message} ---")

            logger.info(
                "Initializing Data Ingestion Training Pipeline."
            )

            # Step 1: Load project configuration.
            config = ConfigurationManager()

            # Step 2: Get Data Ingestion configuration.
            data_ingestion_config = (
                config.get_data_ingestion_config()
            )

            # Step 3: Initialize Data Ingestion component.
            data_ingestion = DataIngestion(
                config=data_ingestion_config
            )

            # Step 4: Execute Data Ingestion.
            results = (
                data_ingestion.initiate_data_ingestion()
            )

            logger.info(
                "Data Ingestion Training Pipeline completed successfully."
            )

            print(
                "\n--- DATA INGESTION PIPELINE COMPLETED ---"
            )

            return results

        except Exception:

            logger.exception(
                "Data Ingestion Training Pipeline failed."
            )

            raise
