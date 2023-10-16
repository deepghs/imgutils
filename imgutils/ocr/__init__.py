"""
Overview:
    Detect and recognize text in images.

    The models are exported from `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_, hosted on
    `huggingface - deepghs/paddleocr <https://huggingface.co/deepghs/paddleocr/tree/main>`_.

    .. image:: ocr_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the text detection models:

    .. image:: ocr_det_benchmark.plot.py.svg
        :align: center

    and an overall benchmark of all the available text recognition models:

    .. image:: ocr_rec_benchmark.plot.py.svg
        :align: center

"""
from .entry import detect_text_with_ocr, ocr, list_det_models, list_rec_models
