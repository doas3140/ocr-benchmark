
from .benchmark import BenchmarkInterface
from .thresholds import togray, threshold, vyto_thresh
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
import numpy as np
import cv2
from PIL import Image
import pytesseract

def img2string(img):
    config = '--psm 6 -c tessedit_char_whitelist=0123456789 --oem 1 -l eng --dpi 300'
    return pytesseract.image_to_string(img, config=config)

class PytesseractSingle(BenchmarkInterface):
    def __init__(self):
        super().__init__()

    def _preprocessing(self, images):
        return [threshold(im, 80, cv2.THRESH_BINARY) for im in images]

    def _images2strings(self, images):
        images = self._preprocessing(images)
        return [img2string(im) for im in images]

    def chars2strings(self, images):
        return self._images2strings(images)
    
    def numbers2strings(self, images):
        return self._images2strings(images)

    def horizontal_numbers2strings(self, images):
        return self._images2strings(images)

    def vertical_numbers2strings(self, images):
        return self._images2strings(images)