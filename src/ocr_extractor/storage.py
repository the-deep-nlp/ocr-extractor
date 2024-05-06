import io
import logging
from typing import Optional
import cv2
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

logging.getLogger().setLevel(logging.INFO)

class StorageHandler:
    def __init__(
        self,
        use_s3: bool,
        bucket_name: str,
        bucket_key: str,
        aws_region_name: str
    ) -> None:
        self.use_s3 = use_s3
        self.bucket_name = bucket_name
        self.bucket_key = bucket_key
        self.s3_client = boto3.client(
            "s3",
            region_name=aws_region_name,
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

    def get_s3_link_or_local_path_for_image(
        self,
        image_data,
        file_extension,
        filename: str
    ):
        """ Store image file in s3 or local disk and returns the path to that file """
        mapper = {
            ".png": "image/png",
            ".jpg": "image/jpg",
            ".jpeg": "image/jpg",
            ".gif": "image/gif"
        }
        file_extension = file_extension.group() if file_extension else ".jpg"
        file_ext = mapper.get(file_extension, "image/jpg") if file_extension else "image/jpg"
        is_success, img_encoded_data = cv2.imencode(file_extension, image_data)
        io_buf = io.BytesIO(img_encoded_data)

        if self.bucket_key:
            merged_bucket_key = f"{self.bucket_key}/{filename}{file_extension}"
        else:
            merged_bucket_key = f"{filename}{file_extension}"

        try:
            self.s3_client.upload_fileobj(
                io_buf,
                Bucket=self.bucket_name,
                Key=merged_bucket_key,
                ExtraArgs={"ContentType": file_ext}
            )
            logging.info("Successfully uploaded the image to s3.")
        except ClientError as cexc:
            logging.info("Error while uploading the image. %s", str(cexc))
            return None
        generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
        return generated_url

    def get_s3_link_or_local_path_for_file(
        self,
        src_filename,
        filename: str,
        file_ext: str="xlsx"
    ):
        """ Store excel file in s3 or local disk and returns the path to that file """
        if self.bucket_key:
            merged_bucket_key = f"{self.bucket_key}/{filename}.{file_ext}"
        else:
            merged_bucket_key = f"{filename}.{file_ext}"
        try:
            self.s3_client.upload_file(
                src_filename,
                Bucket=self.bucket_name,
                Key=merged_bucket_key,
                ExtraArgs={"ContentType": "xlsx"}
            )
            logging.info("Successfully uploaded the excel file to s3.")
        except ClientError as cexc:
            logging.info("Error while uploading the image. %s", str(cexc))
            return None
        generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
        return generated_url
        

    def get_s3_link_or_local_path(
        self,
        contents: str,
        filename: str,
        file_ext: str="json"
    ) -> Optional[str]:
        """ Store contents in s3 or local disk and returns the path to that file """
        if all([self.use_s3, self.bucket_name, self.bucket_key, self.s3_client]):
            merged_bucket_key = f"{filename}.{file_ext}"
            contents_bytes = bytes(contents, "utf-8")
            contents_bytes_obj = io.BytesIO(contents_bytes)
            try:
                self.s3_client.upload_fileobj(
                    contents_bytes_obj,
                    self.bucket_name,
                    merged_bucket_key,
                    ExtraArgs={"ContentType": file_ext}
                )
                logging.info("Successfully uploaded the document.")
            except ClientError as cexc:
                logging.info("Error while uploading the document. %s", str(cexc))
                return None
            generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
            return generated_url
        else:
            logging.info("Not enough info for S3 storage. Storing data in the local disk.")
            output_filepath = f"./outputs/{filename}.{file_ext}"
            with open(output_filepath, "w") as file:
                file.write(contents)
            return output_filepath