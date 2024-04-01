import io
import logging
import operator
from typing import Tuple, Optional
import cv2
import pandas as pd
import numpy as np
import layoutparser as lp
from paddleocr import PaddleOCR, PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from pdf2image import convert_from_path


logging.getLogger().setLevel(logging.INFO)


class OCRBase:
    def __init__(self, image_path: str, is_image: bool) -> None:
        self.is_image = is_image
        self.image_cv = None
        self.pdf_pages = None
        self.image_path = image_path
        if self.is_image:
            self.read_image()
        else:
            self.read_pdf_scanned_doc()

    def read_image(self):
        """ Read the image / scanned doc """
        logging.info("Reading the image")
        self.image_cv = cv2.imread(self.image_path)

    def read_pdf_scanned_doc(self):
        """ Read scanned pdf doc """
        logging.info("Reading the pdf file")
        page_imgs = convert_from_path(self.image_path)
        self.pdf_pages = [np.asarray(x) for x in page_imgs]

    def get_dims(self, img: np.ndarray) -> Tuple[float, float]:
        """ Returns the image dimensions """
        return img.shape[0], img.shape[1]

class LayoutParser(OCRBase):
    """ Detects the layout of the scanned docs """
    def __init__(self, image_path: str, is_image: bool) -> None:
        super().__init__(image_path, is_image)

    def process_layout(
        self,
        img,
        threshold: float=0.5,
        label_map: Optional[dict]=None,
        enforce_cpu: bool=True,
        enable_mkldnn: bool=False
    ):
        """ Get the layout from the scanned doc or images """
        lp_model = lp.PaddleDetectionLayoutModel(
            config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
            threshold=threshold,
            label_map=label_map,
            enforce_cpu=enforce_cpu,
            enable_mkldnn=enable_mkldnn
        )

        layout = lp_model.detect(img)

        text_blocks = lp.Layout([b for b in layout if b.type in ["Text", "Title"]])
        table_blocks = lp.Layout([b for b in layout if b.type=="Table"])
        return text_blocks, table_blocks


class OCRProcessor(LayoutParser):
    """ OCR Processor """
    def __init__(self, image_path: str, lang: str='en', is_image: bool=True) -> None:
        super().__init__(image_path, is_image)
        self.ocr = PaddleOCR(lang=lang, recovery=True)
        self.table_engine = PPStructure(lang=lang, recovery=True)

    def table_extraction(self, img):
        tbl_contents = self.table_engine(img)
        height, width, _ = img.shape
        sorted_tbl_layout = sorted_layout_boxes(tbl_contents, width)

        if sorted_tbl_layout:
            tbl_html_contents = sorted_tbl_layout[0]["res"]["html"]
            df_html = pd.read_html(io.StringIO(tbl_html_contents))[0]
            print(df_html)


    def process(self, image, text_b, table_b, page_num=1):
        """ Extracts the contents from images / scans """
        if table_b:
            logging.info("Table Detected")
            table_boxes = table_b.to_dict()
            sorted_boxes = sorted(table_boxes["blocks"], key=operator.itemgetter('y_1', 'x_1'))
            for table_block in sorted_boxes:
                x_1 = int(table_block["x_1"]) - 5
                y_1 = int(table_block["y_1"]) - 5
                x_2 = int(table_block["x_2"]) + 5
                y_2 = int(table_block["y_2"]) + 5
                crop_img = image[y_1:y_2, x_1:x_2]
                self.table_extraction(crop_img)

        if text_b:
            logging.info("Text Box Detected")
            text_boxes = text_b.to_dict()
            sorted_boxes = sorted(text_boxes["blocks"], key=operator.itemgetter('y_1', 'x_1'))
            for text_block in sorted_boxes:
                x_1 = int(text_block["x_1"]) - 5
                y_1 = int(text_block["y_1"]) - 5
                x_2 = int(text_block["x_2"]) + 5
                y_2 = int(text_block["y_2"]) + 5
                crop_img = image[y_1:y_2, x_1:x_2]
                ocr_outputs = self.ocr.ocr(crop_img)

                for output in ocr_outputs:
                    if output:
                        texts = [line[1][0] for line in output]
                        logging.info(" ".join(texts))

    def handler(self):
        """ OCR Processor """
        label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

        if self.is_image:
            text_b, table_b = self.process_layout(self.image_cv, label_map=label_map)
            self.process(self.image_cv, text_b, table_b)
        else: # pdf scanned file
            logging.info(f"Total number of pages: {len(self.pdf_pages)}")
            for idx, pdf_page in enumerate(self.pdf_pages):
                logging.info(f"Scanning page number {idx}")
                text_b, table_b = self.process_layout(pdf_page, label_map=label_map)
                self.process(pdf_page, text_b, table_b)
