
import json
import logging

from pathlib import Path
from time import perf_counter

import pandas as pd

from Handwriting_Verification_Using_Siamese_Neural_Networks.entity.config_entity import DataValidationConfig


logger = logging.getLogger(__name__)


class DataValidation:

    def __init__(self, config: DataValidationConfig):

        self.config = config

        self.root_dir = config.root_dir

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Required columns from Phase 01 ingestion reports.
        self.required_image_columns = [
            "Split",
            "Category",
            "Image_Name",
            "Image_Path",
            "Width",
            "Height",
            "Is_Valid",
            "Error"
        ]

        self.required_xml_columns = [
            "Split",
            "XML_File",
            "Writer_ID",
            "Image_Reference",
            "Is_Valid",
            "Error"
        ]

        self.required_inventory_columns = [
            "Split",
            "Category",
            "Folder_Exists",
            "File_Count"
        ]

        # Validation outputs.
        self.validation_results = []

        self.reference_errors = []

        self.resolved_references = pd.DataFrame()

        self.writer_statistics = pd.DataFrame()

        # Input reports.
        self.image_df = pd.DataFrame()

        self.xml_df = pd.DataFrame()

        self.inventory_df = pd.DataFrame()

        # Supported image formats.
        self.valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        }

    # =====================================================
    # COMMON HELPER METHODS
    # =====================================================

    @staticmethod
    def normalize_boolean(series):

        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    @staticmethod
    def get_basename(value):

        # Normalize Windows and Linux path separators.
        normalized = str(value).strip().replace("\\", "/")

        return normalized.rsplit("/", 1)[-1]

    # =====================================================
    # TASK 1: LOAD INGESTION REPORTS
    # =====================================================

    def load_ingestion_reports(self):

        logger.info("Loading Phase 01 ingestion reports.")

        self.inventory_df = pd.read_csv(
            self.config.inventory_file,
            dtype=str,
            keep_default_na=False
        )

        self.image_df = pd.read_csv(
            self.config.image_inspection_file,
            dtype=str,
            keep_default_na=False
        )

        self.xml_df = pd.read_csv(
            self.config.xml_inspection_file,
            dtype=str,
            keep_default_na=False
        )

        logger.info("Phase 01 ingestion reports loaded successfully.")

    # =====================================================
    # COMMON VALIDATION RESULT
    # =====================================================

    def add_validation_result(
        self,
        check_name,
        status,
        total_records=0,
        failed_records=0,
        message=""
    ):

        result = {

            "check_name": check_name,

            "status": status,

            "total_records": int(total_records),

            "failed_records": int(failed_records),

            "message": message

        }

        self.validation_results.append(result)

        if status == "FAILED":

            logger.error(f"{check_name}: {message}")

        elif status == "WARNING":

            logger.warning(f"{check_name}: {message}")

        else:

            logger.info(f"{check_name}: {message}")

    # =====================================================
    # TASK 2: VALIDATE REPORT SCHEMAS
    # =====================================================

    def validate_report_schemas(self):

        logger.info("Validating ingestion report schemas.")

        report_schemas = {

            "inventory": (
                self.inventory_df,
                self.required_inventory_columns
            ),

            "image_inspection": (
                self.image_df,
                self.required_image_columns
            ),

            "xml_inspection": (
                self.xml_df,
                self.required_xml_columns
            )

        }

        for report_name, (
            dataframe,
            required_columns
        ) in report_schemas.items():

            missing_columns = [
                column
                for column in required_columns
                if column not in dataframe.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"{report_name} is missing columns: "
                    f"{missing_columns}"
                )

        self.add_validation_result(
            check_name="report_schema_validation",
            status="PASSED",
            total_records=3,
            message="All ingestion report schemas are valid."
        )

    # =====================================================
    # TASK 3: VALIDATE DATASET STRUCTURE
    # =====================================================

    def validate_dataset_structure(self):

        logger.info("Validating dataset structure.")

        missing_folders = self.inventory_df.loc[
            self.inventory_df["Folder_Exists"]
            .str.strip()
            .str.lower()
            .isin(["false", "0", "no"])
        ]

        failed_count = len(missing_folders)

        status = (
            "FAILED"
            if failed_count > 0
            else "PASSED"
        )

        self.add_validation_result(
            check_name="dataset_structure_validation",
            status=status,
            total_records=len(self.inventory_df),
            failed_records=failed_count,
            message=(f"Missing category folders: {failed_count}")
        )

    # =====================================================
    # TASK 4: VALIDATE IMAGE INTEGRITY
    # =====================================================

    def validate_image_integrity(self):

        logger.info("Validating image integrity.")

        image_valid = self.normalize_boolean(
            self.image_df["Is_Valid"]
        )

        missing_names = (
            self.image_df["Image_Name"]
            .str.strip()
            .eq("")
        )

        width = pd.to_numeric(
            self.image_df["Width"],
            errors="coerce"
        )

        height = pd.to_numeric(
            self.image_df["Height"],
            errors="coerce"
        )

        invalid_dimensions = (
            width.isna()
            | height.isna()
            | (width <= 0)
            | (height <= 0)
        )

        failed_mask = (
            ~image_valid
            | missing_names
            | invalid_dimensions
        )

        failed_count = int(
            failed_mask.sum()
        )

        status = (
            "FAILED"
            if failed_count > 0
            else "PASSED"
        )

        self.add_validation_result(
            check_name="image_integrity_validation",
            status=status,
            total_records=len(self.image_df),
            failed_records=failed_count,
            message=(
                f"Invalid image records: {failed_count}"
            )
        )

    # =====================================================
    # TASK 5: VALIDATE XML ANNOTATIONS
    # =====================================================

    def validate_xml_annotations(self):

        logger.info("Validating XML annotation records.")

        xml_valid = self.normalize_boolean(
            self.xml_df["Is_Valid"]
        )

        missing_writer_ids = (
            self.xml_df["Writer_ID"]
            .str.strip()
            .eq("")
        )

        missing_image_references = (
            self.xml_df["Image_Reference"]
            .str.strip()
            .eq("")
        )

        failed_xml_count = int((~xml_valid).sum())

        missing_writer_count = int(missing_writer_ids.sum())

        missing_reference_count = int(
            missing_image_references.sum()
        )

        # XML parsing validation.
        self.add_validation_result(
            check_name="xml_parsing_validation",
            status=(
                "FAILED"
                if failed_xml_count > 0
                else "PASSED"
            ),
            total_records=len(self.xml_df),
            failed_records=failed_xml_count,
            message=(f"Invalid XML records: {failed_xml_count}")
        )

        # Writer ID completeness.
        writer_status = (
            "FAILED"
            if (
                missing_writer_count > 0
                and self.config.fail_on_missing_writer_id
            )
            else (
                "WARNING"
                if missing_writer_count > 0
                else "PASSED"
            )
        )

        self.add_validation_result(
            check_name="writer_id_completeness",
            status=writer_status,
            total_records=len(self.xml_df),
            failed_records=missing_writer_count,
            message=(f"Missing writer IDs: {missing_writer_count}")
        )

        # Original XML metadata completeness.
        # This remains a warning even when a fallback
        # can successfully resolve the image.
        self.add_validation_result(
            check_name="xml_reference_completeness",
            status=(
                "WARNING"
                if missing_reference_count > 0
                else "PASSED"
            ),
            total_records=len(self.xml_df),
            failed_records=missing_reference_count,
            message=(
                f"Missing XML image references: "
                f"{missing_reference_count}"
            )
        )

    # =====================================================
    # TASK 6: BUILD VALID IMAGE INDEX
    # =====================================================

    def build_image_index(self):

        image_valid = self.normalize_boolean(
            self.image_df["Is_Valid"]
        )

        valid_images = self.image_df.loc[
            image_valid
        ].copy()

        image_index = {}

        for row in valid_images.itertuples(index=False):

            split = str(row.Split).strip()

            image_name = self.get_basename(row.Image_Name)

            key = (split,image_name)

            image_index.setdefault(
                key, []
            ).append({
                "Image_Name": image_name,
                "Image_Path": str(row.Image_Path)
            })

        return image_index

    # =====================================================
    # TASK 7: RESOLVE XML IMAGE REFERENCES
    # =====================================================

    def resolve_image_reference(
        self,
        split,
        xml_file,
        image_reference,
        image_index
    ):

        """
        Resolve an XML image reference.

        Priority 1:
            Use the original XML image reference.

        Priority 2:
            If the reference is missing, infer the
            corresponding CVL page image filename.

        A match is accepted only when exactly one
        valid image exists in the same split.
        """

        split = str(split).strip()

        xml_name = self.get_basename(
            xml_file
        )

        image_reference = str(
            image_reference
        ).strip()

        # -------------------------------------------------
        # PRIORITY 1: ORIGINAL XML REFERENCE
        # -------------------------------------------------

        if image_reference:

            reference_name = self.get_basename(
                image_reference
            )

            key = (
                split,
                reference_name
            )

            matches = image_index.get(
                key, []
            )

            if len(matches) == 1:

                return {
                    "resolved": True,
                    "image_name": matches[0]["Image_Name"],
                    "image_path": matches[0]["Image_Path"],
                    "resolution_method": "original_reference",
                    "error_type": "",
                    "details": ""
                }

            if len(matches) > 1:

                return {
                    "resolved": False,
                    "image_name": "",
                    "image_path": "",
                    "resolution_method": "",
                    "error_type": "Ambiguous image reference",
                    "details": (
                        f"Multiple valid images match "
                        f"{reference_name}."
                    )
                }

            return {
                "resolved": False,
                "image_name": "",
                "image_path": "",
                "resolution_method": "",
                "error_type": "Missing image",
                "details": (
                    f"Original reference {reference_name} "
                    f"was not found among valid images "
                    f"in split {split}."
                )
            }

        # -------------------------------------------------
        # PRIORITY 2: FILENAME-BASED FALLBACK
        # -------------------------------------------------

        suffix = "_attributes.xml"

        if not xml_name.lower().endswith(suffix):

            return {
                "resolved": False,
                "image_name": "",
                "image_path": "",
                "resolution_method": "",
                "error_type": "Invalid XML filename",
                "details": (
                    "XML filename does not follow the "
                    "expected CVL naming convention."
                )
            }

        image_stem = xml_name[:-len(suffix)]

        # Search supported image extensions.
        candidate_names = [
            image_stem + extension
            for extension in sorted(
                self.valid_extensions
            )
        ]

        fallback_matches = []

        for candidate in candidate_names:

            key = (
                split,
                candidate
            )

            matches = image_index.get(
                key, []
            )

            fallback_matches.extend(matches)

        # Exactly one valid candidate must exist.
        if len(fallback_matches) == 1:

            match = fallback_matches[0]

            return {
                "resolved": True,
                "image_name": match["Image_Name"],
                "image_path": match["Image_Path"],
                "resolution_method": "filename_fallback",
                "error_type": "",
                "details": (
                    "Original XML reference was missing. "
                    "Image was resolved using the CVL "
                    "XML filename convention."
                )
            }

        if len(fallback_matches) > 1:

            return {
                "resolved": False,
                "image_name": "",
                "image_path": "",
                "resolution_method": "",
                "error_type": "Ambiguous fallback reference",
                "details": (
                    f"Multiple valid images match "
                    f"the XML filename stem {image_stem}."
                )
            }

        return {
            "resolved": False,
            "image_name": "",
            "image_path": "",
            "resolution_method": "",
            "error_type": "Unresolved missing reference",
            "details": (
                f"No valid image was found for XML "
                f"filename {xml_name} in split {split}."
            )
        }

    # =====================================================
    # TASK 8: VALIDATE IMAGE REFERENCES
    # =====================================================

    def validate_image_references(self):

        logger.info(
            "Validating XML-to-image references."
        )

        self.reference_errors = []

        resolved_records = []

        image_index = self.build_image_index()

        for row in self.xml_df.itertuples(index=False):

            split = str(row.Split).strip()

            xml_file = str(row.XML_File).strip()

            writer_id = str(row.Writer_ID).strip()

            original_reference = str(
                row.Image_Reference
            ).strip()

            result = self.resolve_image_reference(
                split=split,
                xml_file=xml_file,
                image_reference=original_reference,
                image_index=image_index
            )

            if result["resolved"]:

                resolved_records.append({

                    "Split": split,

                    "XML_File": xml_file,

                    "Writer_ID": writer_id,

                    "Original_Image_Reference": original_reference,

                    "Resolved_Image_Name": result["image_name"],

                    "Resolved_Image_Path": result["image_path"],

                    "Resolution_Method": result["resolution_method"]

                })

            else:

                self.reference_errors.append({

                    "Split": split,

                    "XML_File": xml_file,

                    "Writer_ID": writer_id,

                    "Image_Reference": original_reference,

                    "Error_Type": result["error_type"],

                    "Details": result["details"]

                })

        self.resolved_references = pd.DataFrame(
            resolved_records,
            columns=[
                "Split",
                "XML_File",
                "Writer_ID",
                "Original_Image_Reference",
                "Resolved_Image_Name",
                "Resolved_Image_Path",
                "Resolution_Method"
            ]
        )

        unresolved_count = len(
            self.reference_errors
        )

        recovered_count = int(
            (
                self.resolved_references[
                    "Resolution_Method"
                ] == "filename_fallback"
            ).sum()
        )

        status = (
            "FAILED"
            if (
                unresolved_count > 0
                and self.config.fail_on_missing_reference
            )
            else (
                "WARNING"
                if unresolved_count > 0
                else "PASSED"
            )
        )

        self.add_validation_result(
            check_name="image_reference_validation",
            status=status,
            total_records=len(self.xml_df),
            failed_records=unresolved_count,
            message=(
                f"Unresolved image references: {unresolved_count}; "
                f"successfully resolved: "
                f"{len(self.resolved_references)}; "
                f"recovered using filename fallback: "
                f"{recovered_count}"
            )
        )

        logger.info(
            f"Resolved references: "
            f"{len(self.resolved_references)}"
        )

        logger.info(
            f"Fallback references: {recovered_count}"
        )

    # =====================================================
    # TASK 9: VALIDATE DUPLICATE REFERENCES
    # =====================================================

    def validate_duplicate_references(self):

        logger.info(
            "Checking duplicate resolved image references."
        )

        if self.resolved_references.empty:

            duplicate_count = 0

        else:

            references = self.resolved_references.copy()

            references["Reference_Key"] = (
                references["Split"].astype(str).str.strip()
                + "::"
                + references["Resolved_Image_Name"].apply(
                    self.get_basename
                )
            )

            duplicate_mask = references[
                "Reference_Key"
            ].duplicated(keep=False)

            duplicate_count = int(
                duplicate_mask.sum()
            )

        status = (
            "FAILED"
            if (
                duplicate_count > 0
                and self.config.fail_on_duplicate_reference
            )
            else (
                "WARNING"
                if duplicate_count > 0
                else "PASSED"
            )
        )

        self.add_validation_result(
            check_name="duplicate_reference_validation",
            status=status,
            total_records=len(self.resolved_references),
            failed_records=duplicate_count,
            message=(
                f"Records involved in duplicate references: "
                f"{duplicate_count}"
            )
        )

    # =====================================================
    # TASK 10: CALCULATE WRITER STATISTICS
    # =====================================================

    def calculate_writer_statistics(self):

        logger.info(
            "Calculating writer-level statistics."
        )

        if self.resolved_references.empty:

            self.writer_statistics = pd.DataFrame(
                columns=[
                    "Split",
                    "Writer_ID",
                    "Sample_Count",
                    "XML_Count",
                    "Split_Writer_Count",
                    "Eligible_For_Positive_Pairs"
                ]
            )

            return

        # Only valid XML records should be used.
        xml_valid = self.normalize_boolean(
            self.xml_df["Is_Valid"]
        )

        valid_xml = self.xml_df.loc[
            xml_valid
        ].copy()

        valid_xml_keys = set(
            zip(
                valid_xml["Split"].astype(str).str.strip(),
                valid_xml["XML_File"].apply(self.get_basename)
            )
        )

        references = self.resolved_references.copy()

        references = references.loc[
            references["Writer_ID"].astype(str).str.strip().ne("")
        ].copy()

        references = references.loc[
            [
                (str(split).strip(), self.get_basename(xml_file))
                in valid_xml_keys
                for split, xml_file in zip(
                    references["Split"],
                    references["XML_File"]
                )
            ]
        ].copy()

        references["Image_Key"] = (
            references["Split"].astype(str).str.strip()
            + "::"
            + references["Resolved_Image_Name"].apply(
                self.get_basename
            )
        )

        # Avoid counting repeated XML annotations
        # for the same writer and image more than once.
        references = references.drop_duplicates(
            subset=[
                "Split",
                "Writer_ID",
                "Image_Key"
            ]
        )

        writer_statistics = (
            references.groupby(
                ["Split", "Writer_ID"]
            )
            .agg(
                Sample_Count=(
                    "Image_Key",
                    "nunique"
                ),
                XML_Count=(
                    "XML_File",
                    "nunique"
                )
            )
            .reset_index()
        )

        writer_statistics["Split_Writer_Count"] = (
            writer_statistics.groupby("Split")[
                "Writer_ID"
            ].transform("nunique")
        )

        # Basic eligibility heuristic only.
        writer_statistics[
            "Eligible_For_Positive_Pairs"
        ] = (
            writer_statistics["Sample_Count"] >= 2
        )

        self.writer_statistics = writer_statistics

        logger.info(
            f"Writer statistics calculated for "
            f"{len(writer_statistics)} split-writer groups."
        )

    # =====================================================
    # TASK 11: VALIDATE WRITER SPLIT OVERLAP
    # =====================================================

    def validate_writer_split_overlap(self):

        logger.info(
            "Checking writer overlap between splits."
        )

        train_writers = set(
            self.xml_df.loc[
                self.xml_df["Split"].str.lower().eq("train"),
                "Writer_ID"
            ].astype(str).str.strip()
        )

        test_writers = set(
            self.xml_df.loc[
                self.xml_df["Split"].str.lower().eq("test"),
                "Writer_ID"
            ].astype(str).str.strip()
        )

        train_writers.discard("")
        test_writers.discard("")

        overlapping_writers = (
            train_writers & test_writers
        )

        overlap_count = len(
            overlapping_writers
        )

        status = (
            "WARNING"
            if overlap_count > 0
            else "PASSED"
        )

        self.add_validation_result(
            check_name="writer_split_overlap",
            status=status,
            total_records=(
                len(train_writers) + len(test_writers)
            ),
            failed_records=overlap_count,
            message=(
                f"Overlapping writers: "
                f"{sorted(overlapping_writers)}"
            )
        )

    # =====================================================
    # TASK 12: SAVE VALIDATION REPORTS
    # =====================================================

    def save_validation_reports(self):

        logger.info(
            "Saving data validation reports."
        )

        # -------------------------------------------------
        # Save writer statistics.
        # -------------------------------------------------

        self.writer_statistics.to_csv(
            self.config.writer_statistics_path,
            index=False
        )

        # -------------------------------------------------
        # Save unresolved image reference errors.
        # -------------------------------------------------

        reference_error_columns = [
            "Split",
            "XML_File",
            "Writer_ID",
            "Image_Reference",
            "Error_Type",
            "Details"
        ]

        reference_errors_df = pd.DataFrame(
            self.reference_errors,
            columns=reference_error_columns
        )

        reference_errors_df.to_csv(
            self.config.image_reference_errors_path,
            index=False
        )

        # -------------------------------------------------
        # Save all resolved image references.
        # -------------------------------------------------

        resolved_reference_path = (
            self.root_dir / "resolved_image_references.csv"
        )

        self.resolved_references.to_csv(
            resolved_reference_path,
            index=False
        )

        # -------------------------------------------------
        # Calculate validation status.
        # -------------------------------------------------

        failed_checks = [
            result
            for result in self.validation_results
            if result["status"] == "FAILED"
        ]

        warning_checks = [
            result
            for result in self.validation_results
            if result["status"] == "WARNING"
        ]

        passed_checks = [
            result
            for result in self.validation_results
            if result["status"] == "PASSED"
        ]

        if failed_checks:

            overall_status = "FAILED"

        elif warning_checks:

            overall_status = "REQUIRES_REVIEW"

        else:

            overall_status = "PASSED"

        # -------------------------------------------------
        # Detailed validation report.
        # -------------------------------------------------

        validation_report = {

            "dataset": "CVL",

            "validation_status": overall_status,

            "total_checks": len(self.validation_results),

            "passed_checks": len(passed_checks),

            "failed_checks": len(failed_checks),

            "warning_checks": len(warning_checks),

            "checks": self.validation_results

        }

        # -------------------------------------------------
        # Validation summary.
        # -------------------------------------------------

        validation_summary = {

            "dataset": "CVL",

            "total_images": len(self.image_df),

            "valid_images": int(
                self.normalize_boolean(
                    self.image_df["Is_Valid"]
                ).sum()
            ),

            "total_xml_files": len(self.xml_df),

            "valid_xml_files": int(
                self.normalize_boolean(
                    self.xml_df["Is_Valid"]
                ).sum()
            ),

            "unique_writers": (
                int(
                    self.writer_statistics["Writer_ID"].nunique()
                )
                if not self.writer_statistics.empty
                else 0
            ),

            "resolved_image_references": len(
                self.resolved_references
            ),

            "fallback_resolved_references": int(
                (
                    self.resolved_references[
                        "Resolution_Method"
                    ] == "filename_fallback"
                ).sum()
            ),

            "unresolved_image_references": len(
                self.reference_errors
            ),

            "image_reference_errors": len(
                self.reference_errors
            ),

            "total_checks": len(self.validation_results),

            "passed_checks": len(passed_checks),

            "failed_checks": len(failed_checks),

            "warning_checks": len(warning_checks),

            "validation_status": overall_status

        }

        # -------------------------------------------------
        # Save JSON reports.
        # -------------------------------------------------

        with open(
            self.config.validation_report_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                validation_report,
                file,
                indent=4,
                ensure_ascii=False
            )

        with open(
            self.config.validation_summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                validation_summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            f"Validation status: {overall_status}"
        )

        logger.info(
            "All validation reports saved successfully."
        )

        return validation_summary

    # =====================================================
    # TASK 13: EXECUTE COMPLETE VALIDATION PIPELINE
    # =====================================================

    def initiate_data_validation(self):

        start_time = perf_counter()

        logger.info(
            "STARTING WRITERLENS DATA VALIDATION"
        )

        # Step 1: Load Phase 01 reports.
        self.load_ingestion_reports()

        # Step 2: Validate report schemas.
        self.validate_report_schemas()

        # Step 3: Validate dataset structure.
        self.validate_dataset_structure()

        # Step 4: Validate image integrity.
        self.validate_image_integrity()

        # Step 5: Validate XML annotations.
        self.validate_xml_annotations()

        # Step 6: Resolve XML image references.
        self.validate_image_references()

        # Step 7: Check duplicate resolved references.
        self.validate_duplicate_references()

        # Step 8: Calculate writer statistics.
        self.calculate_writer_statistics()

        # Step 9: Optional writer overlap check.
        if self.config.check_writer_split_overlap:

            self.validate_writer_split_overlap()

        # Step 10: Save all validation reports.
        summary = self.save_validation_reports()

        elapsed_time = perf_counter() - start_time

        summary["execution_time_seconds"] = round(
            elapsed_time,
            2
        )

        # Update summary JSON with execution time.
        with open(
            self.config.validation_summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            f"Data validation completed in "
            f"{elapsed_time:.2f} seconds."
        )

        logger.info(
            f"Final status: {summary['validation_status']}"
        )

        # Save reports before raising an exception.
        if (
            self.config.strict_mode
            and summary["failed_checks"] > 0
        ):

            raise ValueError(
                "Data validation failed. "
                "Check validation_report.json."
            )

        return summary
