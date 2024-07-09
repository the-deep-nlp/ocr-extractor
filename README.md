# OCR Extractor

This OCR module is intended to retrieve the texts, tables from the images or pdf files using Deep Learning Models developed by Paddle OCR (https://github.com/PaddlePaddle/PaddleOCR).
This library employs several models for detection, recognition, layout parsing, just to name a few.
Based on this library, we wrote a wrapper to process the documents.

The OCRProcessor class can be used as follows:

```
ocr_processor = OCRProcessor(  
    lang: str  -> The language of the document (e.g. 'en')  
    precision: str  -> 'fp16' Use lower value to optimize the floating point used during processing  
    extraction_type: enum[int] (1 - 4)  
    show_log: boolean -> To show the log or not (default: False)  
    layout: boolean  -> Whether to processor the layout of the documents (default: True)  
    use_s3: boolean -> To use s3 for storage (default: False)  
    s3_bucket_name: str -> Name of the bucket in s3 if use_s3 is True (default: None)  
    s3_bucket_key: str -> Key name in s3 if use_s3 is True (default: None)  
    process_text: boolean -> To extract text contents from the images / scanned document (default: True)  
    process_table: boolean -> To extract the tables from the images / pdf / scanned document (default: True)  
    aws_region_name: str -> AWS region (default: 'us-east-1')  
    models_base_path: str -> local location of the models  (default: '/ocr/models/')
)

ocr_processor.load_file(
    file_path: str -> File path which needs to be processed,
    is_image: boolean -> Whether the used file is an image (True) or a pdf (False)
)

result = await ocr_processer.handler()
```

Since few methods used are co-routines in this library, you need to use `async` to execute the co-routine, otherwise co-routine objects are only returned without being processed.

The `result` object is a dictionary consisting of few key-value pairs:  
```
{
    "text": list[objects],
    "image": list[objects],
    "table": list[object]
}
```

The `text` value is a list of objects whose format is:  
```
{
    'page_number': int,
    'order': int,
    'content': str
}
```

The `image` value is a list of objects whose structure is:
```
{
    'page_number': int,
    'images': list[str]  # The strings are the urls
}
```

The `table` value is a list of objects whose structure is:
```
{
    'page_number': int,
    'order': int,
    'content_link': str (url),
    'image_link': str (url)
}
```