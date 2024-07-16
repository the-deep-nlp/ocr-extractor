import os
import logging
import re
import random
from pathlib import Path, PosixPath
from enum import Enum
from typing import Tuple, Generator, Optional
from PIL import Image

import fitz
import numpy as np
import aiofiles
from paddleocr import PPStructure
from html2excel import ExcelParser
from ocr_extractor.storage import StorageHandler
from ocr_extractor.utils import get_ocr_models

from ocr_extractor.utils import filter_image_by_size

logging.getLogger().setLevel(logging.INFO)

class ExtractionType(Enum):
    """ Content Types Extraction from documents / images """
    TEXT_ONLY = 1
    TABLE_ONLY = 2
    TEXT_AND_TABLE = 3
    IMAGE_AND_TABLE = 4
    IMAGE_TABLE_COORDINATES = 5
    ALL = 6


class OCRBase:
    """ OCR module for extracting texts and tables """
    def __init__(self):
        self.file_path = None
        self.is_image = True
        self.img_file_extension = None
        self.image_data = None
        self.pdf_pages = None

    def load_file(self, file_path: str, is_image: bool):
        """ Set the file path """
        self.file_path = file_path
        self.is_image = is_image
        self.img_file_extension = re.search(
            "(?i)\.(jpg|png|jpeg|gif)$",
            file_path
        )

    def read_image(self) -> Optional[np.ndarray]:
        """ Read the image / scanned doc """
        logging.info("Reading the image")
        if os.path.isfile(self.file_path):
            img = Image.open(self.file_path).convert('RGB')
            return np.array(img)
        logging.warning("Cannot open the image file %s", self.file_path)
        return None

    def read_pdf_scanned_doc(self, zoom: tuple=(2,2)) -> Optional[Generator]:
        """ Read scanned pdf doc """
        logging.info("Reading the pdf file")
        if os.path.isfile(self.file_path):
            with fitz.open(self.file_path) as pdf_file:
                for pg in range(0, pdf_file.page_count):
                    page = pdf_file[pg]
                    mat = fitz.Matrix(*zoom)
                    pm = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    yield pg, np.array(img)
        else:
            logging.warning("Cannot open the pdf file %s", self.file_path)
            return None

    def get_dims(self, img: np.ndarray) -> Tuple[float, float]:
        """ Returns the image dimensions """
        return img.shape[0], img.shape[1]

class OCRProcessor(OCRBase):
    """ OCR Processor """
    def __init__(
        self,
        lang: str="en",
        precision: str="fp16",
        extraction_type: int=ExtractionType.TEXT_AND_TABLE.value,
        show_log: bool=False,
        layout: bool=True,
        use_s3: bool=False,
        s3_bucket_name: str=None,
        s3_bucket_key: str=None,
        process_text: bool=True,
        process_table: bool=True,
        aws_region_name: str="us-east-1",
        models_base_path: PosixPath=Path("/tmp/models"),
        **kwargs
    ) -> None:
        """
        Extract contents from the document using OCR.

        file_path: Path of the input file
        lang: The language in which the document is written
        is_image: Type of the document
        extraction_type: Either extract text only or table only or both
        show_log: Whether to display the OCR config logs
        layout: Whether to process the layouts of the document
        use_s3: To check if s3 is used as storage
        s3_bucket_name: Name of the bucket in s3
        s3_bucket_key: Name of the key or file in s3
        process_text: Whether to extract the texts from images / scans
        process_table: Whether to extract the tables from images / scans
        aws_region_name: The AWS region to be used.
        """

        super().__init__()
        self.extraction_type = extraction_type

        if self.extraction_type == ExtractionType.TEXT_ONLY.value:
            process_text = True
            process_table = False
        if self.extraction_type == ExtractionType.TABLE_ONLY.value:
            process_text = False
            process_table = True
        if self.extraction_type == ExtractionType.TEXT_AND_TABLE.value:
            process_text = True
            process_table = True
        if self.extraction_type == ExtractionType.IMAGE_AND_TABLE.value:
            process_text = True
            process_table = True

        ocr_models = get_ocr_models(base_path=models_base_path)
        self.ocr_engine = PPStructure(
            show_log=show_log,
            precision=precision,
            table=process_table,
            ocr=process_text,
            lang=lang,
            layout=layout,
            recovery=True,
            **ocr_models
        )

        self.s3handler = StorageHandler(use_s3, s3_bucket_name, s3_bucket_key, aws_region_name)

        self.final_combined_results = {
            "texts": [],
            "images": [],
            "tables": [],
            "rect_coordinates": []
        }

    async def process(self, image_data: np.ndarray, page_number: int=0):
        """ Extracts the contents from images or scans """

        try:
            results = self.ocr_engine(image_data)
        except Exception as exc:
            logging.error("Exception occurred %s", str(exc), exc_info=True)
            return

        # Sort boxes based on y1 and x1 : bbox top left coordinates
        sorted_results = sorted(results, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        text_sort_number = 0
        table_sort_number = 0
        extracted_images = []
        element_rect_coordinates = []

        for idx, element in enumerate(sorted_results):
            if (element["type"] in ["figure", "table"] and
                self.extraction_type in [
                    ExtractionType.IMAGE_TABLE_COORDINATES.value,
                    ExtractionType.ALL.value
                ]
            ):
                element_rect_coordinates.append({
                    "page_number": page_number,
                    "type": element["type"],
                    "bbox": element["bbox"]
                })
            if (element["type"] in ["figure"] and
                self.extraction_type in [
                    ExtractionType.IMAGE_AND_TABLE.value,
                    ExtractionType.ALL.value
                ]
            ):
                if filter_image_by_size(element["img"]):
                    image_link = self.s3handler.get_s3_link_or_local_path_for_image(
                        element["img"],
                        self.img_file_extension,
                        f"{page_number}_{idx}",
                        "images"
                    )
                    extracted_images.append(str(image_link))

            if (element["type"] in ["text", "figure"] and
                self.extraction_type in [
                    ExtractionType.TEXT_ONLY.value,
                    ExtractionType.IMAGE_AND_TABLE.value,
                    ExtractionType.ALL.value
                ]
            ):
                texts = ""
                for t in element.get("res", []):
                    texts += t["text"] + " "
                self.final_combined_results["texts"].append({
                    "page_number": page_number,
                    "order": text_sort_number,
                    "content": texts.strip()
                })
                text_sort_number += 1
            if (element["type"] == "table" and
                self.extraction_type in [
                    ExtractionType.TABLE_ONLY.value,
                    ExtractionType.TEXT_AND_TABLE.value,
                    ExtractionType.IMAGE_AND_TABLE.value,
                    ExtractionType.ALL.value
                ]
            ):
                if "html" in element.get("res", []):
                    tbl_html_contents = element["res"]["html"]
                    async with aiofiles.tempfile.NamedTemporaryFile("wb+") as tempf:
                        await tempf.write(bytes(tbl_html_contents, "utf-8"))
                        await tempf.seek(0)
                        excel_parser = ExcelParser(tempf.name)
                        temp_filepath_output = Path(f"/tmp/excel_{random.randint(100, 10_000)}_tempfile.xlsx")
                        excel_parser.to_excel(temp_filepath_output)
                    content_link = self.s3handler.get_s3_link_or_local_path_for_file(
                        temp_filepath_output,
                        f"{page_number}_{table_sort_number}",
                        file_ext="xlsx"
                    )
                    image_link = self.s3handler.get_s3_link_or_local_path_for_image(
                        element["img"],
                        self.img_file_extension,
                        f"{page_number}_{table_sort_number}",
                        "tables"
                    )
                    self.final_combined_results["tables"].append({
                        "page_number": page_number,
                        "order": table_sort_number,
                        "content_link": str(content_link),
                        "image_link": str(image_link)
                    })
                    table_sort_number += 1
                else:
                    logging.warning("Table was detected but contents could not be retrived.")
        if extracted_images:
            self.final_combined_results["images"].append({
                "page_number": page_number,
                "images": extracted_images
            })
        if element_rect_coordinates:
            self.final_combined_results["rect_coordinates"].append(element_rect_coordinates)

    async def handler(self) -> dict:
        """ OCR handler for image or pdf file """
        self.final_combined_results = {
            "texts": [],
            "images": [],
            "tables": [],
            "rect_coordinates": []
        }

        if self.is_image:
            image_data = self.read_image()
            await self.process(image_data)
        else:
            for page_num, image_data in self.read_pdf_scanned_doc():
                logging.info("Processing page number %s.", page_num)
                await self.process(image_data=image_data, page_number=page_num)

        return self.final_combined_results
