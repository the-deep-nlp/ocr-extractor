import os
import io
import logging
from pathlib import Path, PosixPath
from typing import Optional, Union
from PIL import Image
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

logging.getLogger().setLevel(logging.INFO)

class StorageHandler:
    """ Store images / files in the s3 or local disk """
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
        filename: str,
        dir_type: str
    ) -> Union[PosixPath, str]:
        """ Store image file in s3 or local disk and returns the path to that file """
        mapper = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif"
        }
        file_extension = file_extension.group(1) if file_extension else "jpeg"
        if file_extension == "jpg":
            file_extension = "jpeg"
        image_content_type = mapper[file_extension] if file_extension else "image/jpeg"

        img = Image.fromarray(image_data)
        buffer = io.BytesIO()

        img.save(buffer, format=file_extension)
        buffer.seek(0,0)

        if all([self.use_s3, self.bucket_name, self.bucket_key, self.s3_client]):
            merged_bucket_key = f"{self.bucket_key}/{dir_type}/{filename}.{file_extension}"
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=merged_bucket_key,
                    Body=buffer,
                    ContentType=image_content_type
                )
                logging.info("Successfully uploaded the image to s3.")
            except ClientError as cexc:
                logging.info("Error while uploading the image. %s", str(cexc))
                return None
            generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
            return generated_url

        fpath = Path("/tmp/images")
        os.makedirs(fpath, exist_ok=True)
        img_fpath = fpath / f"{filename}.{file_extension}"
        img = Image.fromarray(image_data)
        img.save(img_fpath)
        return img_fpath

    def get_s3_link_or_local_path_for_file(
        self,
        src_filename,
        filename: str,
        file_ext: str="xlsx",
        content_type: str="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        """ Store excel file in s3 or local disk and returns the path to that file """
        if all([self.use_s3, self.bucket_name, self.bucket_key, self.s3_client]):
            merged_bucket_key = f"{self.bucket_key}/{filename}.{file_ext}"
            try:
                self.s3_client.upload_file(
                    src_filename,
                    Bucket=self.bucket_name,
                    Key=merged_bucket_key,
                    ExtraArgs={"ContentType": content_type}
                )
                logging.info("Successfully uploaded the excel file to s3.")
            except ClientError as cexc:
                logging.info("Error while uploading the image. %s", str(cexc))
                return None
            generated_url = self.generate_presigned_url(bucket_key=merged_bucket_key)
            return generated_url

        logging.info("Not enough info for S3 storage. Using the local disk.")
        return src_filename
