import io
import logging
import operator
from typing import Tuple, Optional
import cv2
import pandas as pd
import numpy as np
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import layoutparser as lp
from paddleocr import PaddleOCR, PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from pdf2image import convert_from_path


logging.getLogger().setLevel(logging.INFO)


class OCRBase:
    def __init__(self, file_path: str, is_image: bool) -> None:
        self.is_image = is_image
        self.image_cv = None
        self.pdf_pages = None
        self.file_path = file_path
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
        threshold: float=0.5,
        label_map: Optional[dict]=None,
        enforce_cpu: bool=True,
        enable_mkldnn: bool=False
    ) -> Tuple[lp.Layout, lp.Layout]:
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
    def __init__(
        self,
        file_path: str,
        lang: str='en',
        is_image: bool=True,
        use_s3: bool=False,
        s3_bucket_name: str=None,
        s3_bucket_key: str=None
    ) -> None:
        super().__init__(file_path, is_image)
        self.ocr = PaddleOCR(lang=lang, recovery=True)
        self.table_engine = PPStructure(lang=lang, recovery=True)
        self.s3handler = StorageHandler(use_s3, s3_bucket_name, s3_bucket_key)
        

    def table_extraction(self, img) -> Optional[str]:
        """ Extract table contents and return the dataframe """
        tbl_contents = self.table_engine(img)
        height, width, _ = img.shape
        sorted_tbl_layout = sorted_layout_boxes(tbl_contents, width)
        try:
            if sorted_tbl_layout:
                try:
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
                    link = self.s3handler.get_s3_link_or_local_path(tbl_html_contents, f"{page_num}_{idx}")
                    tables_lst.append({
                        "page_number": page_num,
                        "order": idx,
                        "content_link": link  # can be None if error during url generation
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
        results = []
        if self.is_image:
            text_b, table_b = self.process_layout(self.image_cv, label_map=label_map)
            result = self.process(self.image_cv, text_b, table_b)
            results.append(result)
        else: # pdf scanned file
            logging.info("Total number of pages: %s", len(self.pdf_pages))
            for idx, pdf_page in enumerate(self.pdf_pages):
                logging.info("Scanning page number: %s", idx)
                text_b, table_b = self.process_layout(pdf_page, label_map=label_map)
                result = self.process(pdf_page, text_b, table_b, page_num=idx)
                results.append(result)
        return results


class StorageHandler:
    def __init__(
        self,
        use_s3: bool,
        bucket_name: str,
        bucket_key: str
    ) -> None:
        self.use_s3 = use_s3
        self.bucket_name = bucket_name
        self.bucket_key = bucket_key
        self.s3_client = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"}
            )
        )

    def generate_presigned_url(
        self,
        bucket_key: str,
        signed_url_expiry_secs: int=86400
    ) -> Optional[str]:
        """
        Generates a presigned url of the file stored in s3
        """
        try:
            url = self.s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": bucket_key
                },
                ExpiresIn=signed_url_expiry_secs
            )
        except ClientError as cexc:
            logging.error("Error while generating presigned url %s", cexc)
            return None
        return url

    def get_s3_link_or_local_path(
        self,
        html_contents: str,
        tbl_filename: str
    ) -> Optional[str]:
        """ Store contents in s3 or local disk and returns the path to that file """
        if all([self.use_s3, self.bucket_name, self.bucket_key, self.s3_client]):
            merged_bucket_key = f"{self.bucket_key}/{tbl_filename}.html"
            html_contents_bytes = bytes(html_contents, "utf-8")
            html_contents_bytes_obj = io.BytesIO(html_contents_bytes)
            try:
                self.s3_client.upload_fileobj(
                    html_contents_bytes_obj,
                    self.bucket_key,
                    f"{tbl_filename}.html",
                    ExtraArgs={"ContentType": "html"}
                )
            # csv_buf = io.StringIO()
            # df.to_csv(csv_buf, header=True, index=False)
            # csv_buf.seek(0)
            # try:
            #     self.s3_client.put_object(
            #         Bucket=self.bucket_name,
            #         Body=csv_buf.getvalue(),
            #         Key=merged_bucket_key,
            #         ContentType="html"
            #     )
            #     logging.info("Successfully uploaded the document.")
            except ClientError as cexc:
                logging.info("Error while uploading the document. %s", str(cexc))
                return None
            generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
            return generated_url
        else:
            logging.info("Not enough info for S3 storage. Storing data in the local disk.")
            output_filepath = f"./outputs/{tbl_filename}.html"
            with open(output_filepath, "w") as file:
                file.write(html_contents)
            #df.to_csv(output_filepath, header=True, index=False)
            return output_filepath
