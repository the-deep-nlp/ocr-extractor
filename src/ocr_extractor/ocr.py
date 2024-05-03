import logging
import operator
import re
import tempfile
import random
from enum import Enum
from typing import Tuple, Optional
import cv2

import numpy as np

import layoutparser as lp
from paddleocr import PaddleOCR, PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from pdf2image import convert_from_path
from html2excel import ExcelParser

from ocr_extractor.storage import StorageHandler


logging.getLogger().setLevel(logging.INFO)

class ExtractionType(Enum):
    TEXT_ONLY = 1
    TABLE_ONLY = 2
    TEXT_AND_TABLE = 3


class OCRBase:
    def __init__(self, file_path: str, is_image: bool) -> None:
        self.is_image = is_image
        self.image_cv = None
        self.pdf_pages = None
        self.file_path = file_path
        self.file_extension = re.search(
            "(?i)\.(jpg|png|jpeg|gif)$",
            file_path
        )
        if self.is_image:
            self.read_image()
        else:
            self.read_pdf_scanned_doc()

    def read_image(self) -> None:
        """ Read the image / scanned doc """
        logging.info("Reading the image")
        self.image_cv = cv2.imread(self.file_path)

    def read_pdf_scanned_doc(self) -> None:
        """ Read scanned pdf doc """
        logging.info("Reading the pdf file")
        page_imgs = convert_from_path(self.file_path)
        self.pdf_pages = [np.asarray(x) for x in page_imgs]

    def get_dims(self, img: np.ndarray) -> Tuple[float, float]:
        """ Returns the image dimensions """
        return img.shape[0], img.shape[1]

class LayoutParser(OCRBase):
    """ Detects the layout of the scanned docs """
    def __init__(self, file_path: str, is_image: bool) -> None:
        super().__init__(file_path, is_image)

    def process_layout(
        self,
        img,
        extraction_type,
        threshold: float=0.5,
        label_map: Optional[dict]=None,
        enforce_cpu: bool=True,
        enable_mkldnn: bool=False
    ) -> Tuple[lp.Layout, lp.Layout]:
        """ Get the layout from the scanned doc or images """
        lp_model = lp.PaddleDetectionLayoutModel(
            #config_path="lp://test/PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
            model_path="/ocr/models/ppyolov2_r50vd_dcn_365e_publaynet",
            threshold=threshold,
            label_map=label_map,
            enforce_cpu=enforce_cpu,
            enable_mkldnn=enable_mkldnn
        )

        layout = lp_model.detect(img)
        
        text_blocks = lp.Layout([])
        table_blocks = lp.Layout([])
        if extraction_type in [ExtractionType.TEXT_ONLY.value, ExtractionType.TEXT_AND_TABLE.value]:
            text_blocks = lp.Layout([b for b in layout if b.type in ["Text", "Title"]])
        if extraction_type in [ExtractionType.TABLE_ONLY.value, ExtractionType.TEXT_AND_TABLE.value]:
            table_blocks = lp.Layout([b for b in layout if b.type=="Table"])
        return text_blocks, table_blocks


class OCRProcessor(LayoutParser):
    """ OCR Processor """
    def __init__(
        self,
        file_path: str,
        lang: str='en',
        is_image: bool=True,
        extraction_type: int=ExtractionType.TEXT_AND_TABLE.value,
        use_s3: bool=False,
        s3_bucket_name: str=None,
        s3_bucket_key: str=None,
        aws_region_name: str="us-east-1"
    ) -> None:
        """
        Extract contents from the document using OCR.

        file_path: Path of the input file
        lang: The language in which the document is written
        is_image: Type of the document
        extraction_type: Either extract text only or table only or both
        use_s3: To check if s3 is used as storage
        s3_bucket_name: Name of the bucket in s3
        s3_bucket_key: Name of the key or file in s3
        aws_region_name: The AWS region to be used.
        """
        super().__init__(file_path, is_image)
        self.ocr = PaddleOCR(
            lang=lang,
            recovery=True,
            det_model_dir="/ocr/models/en_PP-OCRv3_det_infer",
            table_model_dir="/ocr/models/en_ppstructure_mobile_v2.0_SLANet_infer",
            rec_model_dir="/ocr/models/en_PP-OCRv4_rec_infer",
            layout_model_dir="/ocr/models/picodet_lcnet_x1_0_fgd_layout_infer"
        )
        self.extraction_type = extraction_type
        self.table_engine = PPStructure(
            lang=lang,
            recovery=True,
            det_model_dir="/ocr/models/en_PP-OCRv3_det_infer",
            table_model_dir="/ocr/models/en_ppstructure_mobile_v2.0_SLANet_infer",
            rec_model_dir="/ocr/models/en_PP-OCRv4_rec_infer",
            layout_model_dir="/ocr/models/picodet_lcnet_x1_0_fgd_layout_infer"
        )
        self.s3handler = StorageHandler(use_s3, s3_bucket_name, s3_bucket_key, aws_region_name)
        

    def table_extraction(self, img) -> Optional[str]:
        """ Extract table contents and return the dataframe """
        tbl_contents = self.table_engine(img)
        height, width, _ = img.shape
        sorted_tbl_layout = sorted_layout_boxes(tbl_contents, width)
        try:
            if sorted_tbl_layout:
                try:
                    if not sorted_tbl_layout[0]["type"] == "table":
                        return None
                    tbl_html_contents = sorted_tbl_layout[0]["res"]["html"]
                    #df_html = pd.read_html(io.StringIO(tbl_html_contents))[0]
                    return tbl_html_contents
                except (KeyError, IndexError):
                    return None
        except (KeyError, IndexError):
            logging.warning("Table contents not found in html format.")
        return None


    def process(self, image, text_b, table_b, page_num=1):
        """ Extracts the contents from images / scans """
        tables_lst = []
        texts_lst = []
        if table_b:
            logging.info("Table Detected")
            table_boxes = table_b.to_dict()
            sorted_boxes = sorted(table_boxes["blocks"], key=operator.itemgetter('y_1', 'x_1'))
            for idx, table_block in enumerate(sorted_boxes):
                x_1 = int(table_block["x_1"]) - 5
                y_1 = int(table_block["y_1"]) - 5
                x_2 = int(table_block["x_2"]) + 5
                y_2 = int(table_block["y_2"]) + 5
                crop_img = image[y_1:y_2, x_1:x_2]
                tbl_html_contents = self.table_extraction(crop_img)
                if tbl_html_contents:
                    tempf = tempfile.NamedTemporaryFile()
                    tempf.write(bytes(tbl_html_contents, "utf-8"))
                    tempf.seek(0)
                    excel_parser = ExcelParser(tempf.name)
                    temp_filepath_output = f"/tmp/{random.randint(100, 1000)}_tempfile.xlsx"
                    excel_parser.to_excel(temp_filepath_output)
                    content_link = self.s3handler.get_s3_link_or_local_path_for_file(temp_filepath_output, f"{page_num}_{idx}", file_ext="xlsx")
                    image_link = self.s3handler.get_s3_link_or_local_path_for_image(image, self.file_extension, f"{page_num}_{idx}")
                    tables_lst.append({
                        "page_number": page_num,
                        "order": idx,
                        "content_link": content_link,
                        "image_link": image_link
                    })

        if text_b:
            logging.info("Text Box Detected")
            text_boxes = text_b.to_dict()
            sorted_boxes = sorted(text_boxes["blocks"], key=operator.itemgetter('y_1', 'x_1'))
            for idx, text_block in enumerate(sorted_boxes):
                x_1 = int(text_block["x_1"]) - 5
                y_1 = int(text_block["y_1"]) - 5
                x_2 = int(text_block["x_2"]) + 5
                y_2 = int(text_block["y_2"]) + 5
                crop_img = image[y_1:y_2, x_1:x_2]
                ocr_outputs = self.ocr.ocr(crop_img)

                for output in ocr_outputs:
                    if output:
                        texts = [line[1][0] for line in output]
                        plain_texts = " ".join(texts)
                        texts_lst.append({
                            "page_number": page_num,
                            "order": idx,
                            "content": plain_texts
                        })
        return texts_lst, tables_lst

    def handler(self) -> list:
        """ OCR Processor """
        label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        text_results = []
        table_results = []
        if self.is_image:
            text_b, table_b = self.process_layout(self.image_cv, self.extraction_type, label_map=label_map)
            texts_lst, tables_lst = self.process(self.image_cv, text_b, table_b)
            text_results.append(texts_lst)
            table_results.append(tables_lst)
        else: # pdf scanned file
            logging.info("Total number of pages: %s", len(self.pdf_pages))
            for idx, pdf_page in enumerate(self.pdf_pages):
                logging.info("Scanning page number: %s", idx+1)
                text_b, table_b = self.process_layout(pdf_page, self.extraction_type, label_map=label_map)
                texts_lst, tables_lst = self.process(pdf_page, text_b, table_b, page_num=idx+1)
                text_results.append(texts_lst)
                table_results.append(tables_lst)
        return text_results, table_results




