# IMPORT LIBRARIES FOR DATA INGESTION COMPONENT

import json
import re
import xml.etree.ElementTree as ET

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from Handwriting_Verification_Using_Siamese_Neural_Networks.entity.config_entity import DataIngestionConfig

from Handwriting_Verification_Using_Siamese_Neural_Networks.logging import logger


# COMPONENTS

class DataIngestion:
    """
    CVL dataset ingestion component.

    Responsibilities:
        1. Validate dataset directories.
        2. Collect recursive file inventory.
        3. Inspect page, line and word images.
        4. Parse XML annotation files.
        5. Extract available XML metadata.
        6. Generate ingestion reports.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.report_dir = self.config.root_dir

    # ---------------------------------------------------------
    # TASK 1: VALIDATE DATASET
    # ---------------------------------------------------------

    def validate_dataset(self):
        required_paths = {
            "Dataset": self.config.dataset_dir,
            "Training": self.config.train_dir,
            "Testing": self.config.test_dir,
        }

        for name, path in required_paths.items():

            if not path.exists():
                raise FileNotFoundError(
                    f"{name} directory not found: {path}"
                )

            if not path.is_dir():
                raise NotADirectoryError(
                    f"{name} path is not a directory: {path}"
                )

        logger.info(
            "CVL dataset directories validated successfully."
        )

    # ---------------------------------------------------------
    # TASK 2: COLLECT RECURSIVE DATASET INVENTORY
    # ---------------------------------------------------------

    def collect_file_inventory(self):
        logger.info("Starting dataset file inventory.")

        inventory = []

        dataset_splits = {
            "train": self.config.train_dir,
            "test": self.config.test_dir,
        }

        categories = [
            *self.config.image_types,
            self.config.annotation_dir,
        ]

        for split_name, split_path in dataset_splits.items():

            for category in categories:

                category_path = split_path / category

                folder_exists = category_path.is_dir()

                if folder_exists:
                    # Recursive discovery includes nested files.
                    files = [
                        path
                        for path in category_path.rglob("*")
                        if path.is_file()
                    ]

                else:
                    files = []

                    logger.warning(
                        f"Category folder not found: {category_path}"
                    )

                total_size_bytes = sum(
                    path.stat().st_size for path in files
                )

                extensions = sorted({
                    path.suffix.lower()
                    for path in files
                    if path.suffix
                })

                inventory.append({
                    "Split": split_name,
                    "Category": category,
                    "Folder_Exists": folder_exists,
                    "File_Count": len(files),
                    "Total_Size_MB": round(
                        total_size_bytes / (1024 ** 2), 2
                    ),
                    "Extensions": ", ".join(extensions),
                })

        inventory_df = pd.DataFrame(inventory)

        logger.info("Dataset inventory completed.")

        return inventory_df

    # ---------------------------------------------------------
    # TASK 3: IMAGE INSPECTION
    # ---------------------------------------------------------

    def inspect_images(self):
        logger.info("Starting image inspection.")

        image_records = []

        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        dataset_splits = {
            "train": self.config.train_dir,
            "test": self.config.test_dir,
        }

        for split_name, split_path in dataset_splits.items():

            for category in self.config.image_types:

                category_path = split_path / category

                if not category_path.is_dir():
                    logger.warning(
                        f"Image category not found: {category_path}"
                    )
                    continue

                # Recursively discover images in nested directories.
                image_files = sorted(
                    path
                    for path in category_path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in image_extensions
                )

                logger.info(
                    f"Inspecting {len(image_files)} images "
                    f"in {split_name}/{category}."
                )

                for image_path in image_files:

                    record = {
                        "Split": split_name,
                        "Category": category,
                        "Image_Name": image_path.name,
                        "Image_Path": str(image_path),
                        "Extension": image_path.suffix.lower(),
                        "Width": None,
                        "Height": None,
                        "Channels": None,
                        "Color_Mode": None,
                        "Image_Format": None,
                        "Is_Valid": False,
                        "Error": None,
                    }

                    try:
                        # First verify image integrity.
                        with Image.open(image_path) as image:
                            image.verify()

                        # Reopen to retrieve image properties.
                        with Image.open(image_path) as image:

                            record["Width"] = image.width
                            record["Height"] = image.height
                            record["Channels"] = len(
                                image.getbands()
                            )
                            record["Color_Mode"] = image.mode
                            record["Image_Format"] = image.format
                            record["Is_Valid"] = True

                    except (
                        UnidentifiedImageError,
                        OSError,
                        ValueError,
                    ) as error:

                        record["Error"] = str(error)

                        logger.warning(
                            f"Invalid image: {image_path} | {error}"
                        )

                    image_records.append(record)

        image_df = pd.DataFrame(image_records)

        valid_count = (
            int(image_df["Is_Valid"].sum())
            if not image_df.empty else 0
        )

        invalid_count = len(image_df) - valid_count

        logger.info(
            f"Image inspection completed. "
            f"Total: {len(image_df)}, "
            f"Valid: {valid_count}, "
            f"Invalid: {invalid_count}"
        )

        return image_df

    # ---------------------------------------------------------
    # TASK 4: XML PARSING
    # ---------------------------------------------------------

    @staticmethod
    def parse_xml_file(xml_path):
        """
        Parse XML without modifying the original source file.

        First attempts parsing directly from original bytes.
        If that fails, tries supported text encodings.
        """

        raw_content = Path(xml_path).read_bytes()

        try:
            return ET.fromstring(raw_content)

        except ET.ParseError:
            pass

        encodings = [
            "utf-8-sig",
            "utf-8",
            "cp1252",
            "latin-1",
        ]

        for encoding in encodings:

            try:
                decoded_content = raw_content.decode(encoding)

                # Normalize the XML declaration when retrying.
                decoded_content = re.sub(
                    r"""(<\?xml[^?]*encoding\s*=\s*["'])[^"']+(["'][^?]*\?>)""",
                    r"\g<1>UTF-8\2",
                    decoded_content,
                    count=1,
                    flags=re.IGNORECASE,
                )

                return ET.fromstring(
                    decoded_content.encode("utf-8")
                )

            except (
                UnicodeDecodeError,
                ET.ParseError,
                ValueError,
            ):
                continue

        raise ValueError(
            f"Unable to parse XML file: {xml_path}"
        )

    # ---------------------------------------------------------
    # TASK 5: XML ANNOTATION INSPECTION
    # ---------------------------------------------------------

    def inspect_xml_annotations(self):
        logger.info("Starting XML annotation inspection.")

        xml_records = []

        dataset_splits = {
            "train": self.config.train_dir,
            "test": self.config.test_dir,
        }

        def get_local_tag(tag):
            """Remove XML namespace from element names."""
            return tag.split("}")[-1]

        for split_name, split_path in dataset_splits.items():

            xml_dir = split_path / self.config.annotation_dir

            if not xml_dir.is_dir():
                logger.warning(
                    f"XML directory not found: {xml_dir}"
                )
                continue

            xml_files = sorted(xml_dir.rglob("*.xml"))

            for xml_path in xml_files:

                record = {
                    "Split": split_name,
                    "XML_File": xml_path.name,
                    "Root_Tag": None,
                    "Writer_ID": None,
                    "Writer_ID_Source": "Filename inference",
                    "Image_Reference": None,
                    "Image_Width": None,
                    "Image_Height": None,
                    "Text_Line_Count": 0,
                    "Region_Count": 0,
                    "Point_Count": 0,
                    "Transcription_Available": False,
                    "Is_Valid": False,
                    "Error": None,
                }

                try:
                    root = self.parse_xml_file(xml_path)

                    record["Root_Tag"] = get_local_tag(root.tag)

                    # Extract writer ID from CVL filename.
                    # Example: 0001-1_attributes.xml -> 0001
                    filename_match = re.match(
                        r"^(\d+)-",
                        xml_path.stem,
                    )

                    if filename_match:
                        record["Writer_ID"] = (
                            filename_match.group(1)
                        )

                    # Extract page-level metadata and annotation counts.
                    for element in root.iter():

                        tag = get_local_tag(element.tag)

                        if tag == "Page":

                            record["Image_Reference"] = (
                                element.attrib.get("imageFilename")
                            )

                            record["Image_Width"] = (
                                element.attrib.get("imageWidth")
                            )

                            record["Image_Height"] = (
                                element.attrib.get("imageHeight")
                            )

                        elif tag == "textlines":

                            record["Text_Line_Count"] += 1

                        elif tag == "AttrRegion":

                            record["Region_Count"] += 1

                        elif tag == "Point":

                            record["Point_Count"] += 1

                        elif tag == "Unicode":

                            if element.text and element.text.strip():
                                record["Transcription_Available"] = True

                    record["Is_Valid"] = True

                except (
                    ET.ParseError,
                    OSError,
                    ValueError,
                ) as error:

                    record["Error"] = str(error)

                    logger.warning(
                        f"XML inspection failed: "
                        f"{xml_path.name} | {error}"
                    )

                xml_records.append(record)

        xml_df = pd.DataFrame(xml_records)

        valid_count = (
            int(xml_df["Is_Valid"].sum())
            if not xml_df.empty else 0
        )

        invalid_count = len(xml_df) - valid_count

        logger.info(
            f"XML inspection completed. "
            f"Total: {len(xml_df)}, "
            f"Valid: {valid_count}, "
            f"Invalid: {invalid_count}"
        )

        return xml_df

    # ---------------------------------------------------------
    # TASK 6: SAVE INGESTION REPORTS
    # ---------------------------------------------------------

    def save_ingestion_reports(
        self,
        inventory_df,
        image_df,
        xml_df,
    ):

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        inventory_csv = (
            self.report_dir / "dataset_inventory.csv"
        )

        image_csv = (
            self.report_dir / "image_inspection.csv"
        )

        xml_csv = (
            self.report_dir / "xml_annotation_inspection.csv"
        )

        # Save CSV reports.
        inventory_df.to_csv(
            inventory_csv,
            index=False,
        )

        image_df.to_csv(
            image_csv,
            index=False,
        )

        xml_df.to_csv(
            xml_csv,
            index=False,
        )

        total_files = int(
            inventory_df["File_Count"].sum()
        )

        total_size_mb = round(
            float(inventory_df["Total_Size_MB"].sum()),
            2,
        )

        valid_images = (
            int(image_df["Is_Valid"].sum())
            if not image_df.empty else 0
        )

        valid_xml = (
            int(xml_df["Is_Valid"].sum())
            if not xml_df.empty else 0
        )

        # Category-wise image counts.
        image_category_summary = {}

        if not image_df.empty:

            image_category_summary = (
                image_df.groupby(
                    ["Split", "Category"]
                )["Image_Name"]
                .count()
                .to_dict()
            )

            image_category_summary = {
                f"{split}_{category}": int(count)
                for (split, category), count
                in image_category_summary.items()
            }

        # Writer ID source is explicitly marked as inferred.
        writer_count = (
            int(xml_df["Writer_ID"].nunique())
            if not xml_df.empty else 0
        )

        summary = {
            "dataset_name": self.config.dataset_name,
            "dataset_path": str(self.config.dataset_dir),

            "dataset_inventory": {
                "total_files": total_files,
                "total_size_mb": total_size_mb,
                "categories": inventory_df.to_dict(
                    orient="records"
                ),
            },

            "image_inspection": {
                "total_images": len(image_df),
                "valid_images": valid_images,
                "invalid_images": len(image_df) - valid_images,
                "category_counts": image_category_summary,
            },

            "xml_inspection": {
                "total_xml_files": len(xml_df),
                "valid_xml_files": valid_xml,
                "invalid_xml_files": len(xml_df) - valid_xml,
                "unique_inferred_writer_ids": writer_count,
                "transcription_available": (
                    bool(xml_df["Transcription_Available"].any())
                    if not xml_df.empty else False
                ),
            },

            "reports": {
                "inventory_csv": inventory_csv.name,
                "image_inspection_csv": image_csv.name,
                "xml_inspection_csv": xml_csv.name,
            },
        }

        summary_json = (
            self.report_dir / "ingestion_summary.json"
        )

        with open(
            summary_json,
            "w",
            encoding="utf-8",
        ) as json_file:

            json.dump(
                summary,
                json_file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "All ingestion reports saved successfully."
        )

        print("\nINGESTION REPORTS SAVED")

        print(f"Inventory: {inventory_csv}")
        print(f"Image inspection: {image_csv}")
        print(f"XML inspection: {xml_csv}")
        print(f"Summary: {summary_json}")

        return summary

    # ---------------------------------------------------------
    # TASK 7: EXECUTE COMPLETE DATA INGESTION
    # ---------------------------------------------------------

    def initiate_data_ingestion(self):

        print("\nSTARTING WRITERLENS DATA INGESTION")

        # Step 1: Validate dataset.
        self.validate_dataset()

        # Step 2: Collect dataset inventory.
        print("\n[1/4] Dataset inventory...")

        inventory_df = self.collect_file_inventory()

        # Step 3: Inspect all image categories.
        print("\n[2/4] Image inspection...")

        image_df = self.inspect_images()

        # Step 4: Inspect XML annotations.
        print("\n[3/4] XML annotation inspection...")

        xml_df = self.inspect_xml_annotations()

        # Step 5: Save reports.
        print("\n[4/4] Saving ingestion reports...")

        summary = self.save_ingestion_reports(
            inventory_df=inventory_df,
            image_df=image_df,
            xml_df=xml_df,
        )

        print("\nDATA INGESTION COMPLETED")

        return {
            "inventory": inventory_df,
            "image_inspection": image_df,
            "xml_inspection": xml_df,
            "summary": summary,
        }
