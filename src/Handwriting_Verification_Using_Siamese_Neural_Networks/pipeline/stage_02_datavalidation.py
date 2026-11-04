from Handwriting_Verification_Using_Siamese_Neural_Networks.config.configuration import ConfigurationManager


from Handwriting_Verification_Using_Siamese_Neural_Networks.logging import logger

from Handwriting_Verification_Using_Siamese_Neural_Networks.components.data_validation import DataValidation

# pipeline 


class DataValidationTrainingPipeline:

    def __init__(self):
        pass


    def main(self):

        try:

            print("\n--- STARTING DATA VALIDATION PIPELINE ---")

            logger.info("Initializing Data Validation Pipeline.")

            # Step 1: Load project configuration.
            config = ConfigurationManager()

            # Step 2: Get Data Validation configuration.
            data_validation_config = (config.get_data_validation_config())

            # Step 3: Initialize Data Validation component.
            data_validation = DataValidation(config=data_validation_config)

            # Step 4: Execute Data Validation.
            results = (data_validation.initiate_data_validation()
                       )

            logger.info("Data Validation Pipeline completed successfully.")

            print("\n--- DATA VALIDATION PIPELINE COMPLETED ---")

            print(f"\nValidation Status: "
                  f"{results['validation_status']}")

            print(f"Passed Checks: {results['passed_checks']}")

            print(f"Failed Checks: {results['failed_checks']}")

            print(f"Warnings: {results['warning_checks']}")

            return results

        except Exception:

            logger.exception("Data Validation Pipeline failed.")

            print("\n--- DATA VALIDATION PIPELINE FAILED ---")

            raise