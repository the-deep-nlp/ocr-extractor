from setuptools import setup, find_packages

PACKAGE_VERSION = "0.1"
MINOR_RELEASE = "0"

setup(
    name="ocr_extractor",
    version=f"{PACKAGE_VERSION}.{MINOR_RELEASE}",
    packages=find_packages(where="src"),  # include all packages under src
    package_dir={"": "src"},
    include_package_data=True,

    install_requires=[
        "bce-python-sdk==0.9.5",
        "paddleocr==2.7.0.3",
        "paddlepaddle==2.6.0",
        "pdf2image==1.17.0",
        "layoutparser@https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl",
        "boto3==1.26.106",
        "html2excel==0.0.6"
        ],

    author="Ranjan Shrestha",
    author_email="",
    description="This package contains scripts to extract data from the scanned documents",
    classifiers=[
        ""
    ]
)
