import os
import logging
import re
import tempfile
import random
from enum import Enum
from typing import Tuple, Generator, Optional

from PIL import Image

import fitz
import numpy as np

from paddleocr import PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from html2excel import ExcelParser

from ocr_extractor.storage import StorageHandler


logging.getLogger().setLevel(logging.INFO)

class ExtractionType(Enum):
    """ Content Types Extraction from documents / images """
    TEXT_ONLY = 1
    TABLE_ONLY = 2
    TEXT_AND_TABLE = 3


class OCRBase:
    """ OCR module for extracting texts and tables """
    def __init__(self, file_path: str, is_image: bool) -> None:
        self.is_image = is_image
        self.image_data = None
        self.pdf_pages = None
        self.file_path = file_path
        self.img_file_extension = re.search(
            "(?i)\.(jpg|png|jpeg|gif)$",
            file_path
        )

    def read_image(self) -> Optional[np.ndarray]:
        """ Read the image / scanned doc """
        logging.info("Reading the image")
        if os.path.isfile(self.file_path):
            return np.array(Image.open(self.file_path))
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
        file_path: str,
        lang: str='en',
        is_image: bool=True,
        extraction_type: int=ExtractionType.TEXT_AND_TABLE.value,
        show_log: bool=True,
        layout: bool=True,
        use_s3: bool=False,
        s3_bucket_name: str=None,
        s3_bucket_key: str=None,
        process_text: bool=True,
        process_table: bool=True,
        aws_region_name: str="us-east-1",
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
   
        super().__init__(file_path, is_image)
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

        self.ocr_engine = PPStructure(
            show_log=show_log,
            table=process_table,
            ocr=process_text,
            lang=lang,
            layout=layout,
            recovery=True,
            **kwargs
        )

        self.s3handler = StorageHandler(use_s3, s3_bucket_name, s3_bucket_key, aws_region_name)

        self.final_combined_results = {
            "text": [],
            "table": []
        }

    def process(self, image_data: np.ndarray, page_number: int=0):
        """ Extracts the contents from images or scans """

        try:
            results = self.ocr_engine(image_data)
        except Exception as exc:
            logging.error(f"Exception occurred {str(exc)}", exc_info=True)
            return # TODO
        
        # Sort boxes based on y1 and x1 : bbox top left coordinates
        sorted_results = sorted(results, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        text_sort_number = 0
        table_sort_number = 0
        for element in sorted_results:
            if (element["type"] == "text" and
                self.extraction_type in [
                    ExtractionType.TEXT_ONLY.value,
                    ExtractionType.TEXT_AND_TABLE.value
                ]
            ):
                texts = ""
                for t in element.get("res", []):
                    texts += t["text"] + " "
                self.final_combined_results[element["type"]].append({
                    "page_number": page_number,
                    "order": text_sort_number,
                    "content": texts.strip()
                })
                text_sort_number += 1
            if (element["type"] == "table" and
                self.extraction_type in [
                    ExtractionType.TABLE_ONLY.value,
                    ExtractionType.TEXT_AND_TABLE.value
                ]
            ):
                if "html" in element.get("res", []):
                    tbl_html_contents = element["res"]["html"]
                    with tempfile.NamedTemporaryFile() as tempf:
                        tempf.write(bytes(tbl_html_contents, "utf-8"))
                        tempf.seek(0)
                        excel_parser = ExcelParser(tempf.name)
                        temp_filepath_output = f"/tmp/{random.randint(100, 1000)}_tempfile.xlsx"
                        excel_parser.to_excel(temp_filepath_output)
                    content_link = self.s3handler.get_s3_link_or_local_path_for_file(
                        temp_filepath_output,
                        f"{page_number}_{table_sort_number}",
                        file_ext="xlsx"
                    )
                    image_link = self.s3handler.get_s3_link_or_local_path_for_image(
                        element["img"],
                        self.img_file_extension,
                        f"{page_number}_{table_sort_number}"
                    )
                    self.final_combined_results[element["type"]].append({
                        "page_number": page_number,
                        "order": table_sort_number,
                        "content_link": content_link,
                        "image_link": image_link
                    })
                    table_sort_number += 1
                else:
                    logging.warning("Table was detected but contents could not be retrived.")

    def handler(self) -> dict:
        """ OCR handler for image or pdf file """

        if self.is_image:
            image_data = self.read_image()
            self.process(image_data)
            logging.info(self.final_combined_results)

        else:
            for page_num, image_data in self.read_pdf_scanned_doc():
                logging.info("Processing page number %s.", page_num)
                self.process(image_data=image_data, page_number=page_num)
            logging.info(self.final_combined_results)
