import os
import tarfile
import logging
from pathlib import Path, PosixPath
from ocr_extractor.wget import download

logging.getLogger().setLevel(logging.INFO)

def get_ocr_models(base_path: PosixPath):
    """ Get OCR models"""
    model_urls = [
        "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
        "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
        "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar"
    ]

    tar_files = [
        "en_PP-OCRv3_det_infer.tar",
        "en_PP-OCRv4_rec_infer.tar",
        "ch_ppocr_mobile_v2.0_cls_infer.tar",
        "en_ppstructure_mobile_v2.0_SLANet_infer.tar",
        "picodet_lcnet_x1_0_fgd_layout_infer.tar",
        "ppyolov2_r50vd_dcn_365e_publaynet.tar"
    ]

    file_paths = {
        "det_model_dir": base_path / "en_PP-OCRv3_det_infer",
        "rec_model_dir": base_path / "en_PP-OCRv4_rec_infer",
        "cls_model_dir": base_path / "ch_ppocr_mobile_v2.0_cls_infer",
        "table_model_dir": base_path / "en_ppstructure_mobile_v2.0_SLANet_infer",
        "layout_model_dir": base_path / "picodet_lcnet_x1_0_fgd_layout_infer"
    }

    if not os.path.exists(base_path):
        os.makedirs(base_path)

        for url in model_urls:
            logging.info("Downloading the model %s", url)
            download(url=url, out=str(base_path))

        for tar_file in tar_files:
            with tarfile.open(base_path / tar_file) as tar:
                tar.extractall(base_path)
    else:
        logging.info("OCR models path exist.")
    return {
        k: str(v) for k, v in file_paths.items()
    }
