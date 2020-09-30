
from .benchmark import BenchmarkInterface
from .thresholds import togray, threshold, vyto_thresh
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
import numpy as np
import cv2
from PIL import Image

def init_api(**kwargs):
    api = PyTessBaseAPI(**kwargs)
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("tessedit_do_invert", "0")
    return api

def img2string(img, api):
    api.SetImage(Image.fromarray(img))
    return api.GetUTF8Text().strip().replace(' ', '')

class TesseractSingle(BenchmarkInterface):
    lstm = False
    def __init__(self):
        super().__init__()
        oem = OEM.LSTM_ONLY if self.lstm else OEM.TESSERACT_ONLY
        lang = 'eng'
        self._line_api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)
        self._char_api = init_api(psm=PSM.SINGLE_CHAR, oem=oem, lang=lang)
        self._hori_api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)
        self._vert_api = init_api(psm=PSM.SINGLE_BLOCK_VERT_TEXT, oem=oem, lang=lang)

    def __del__(self):
        self._line_api.End()
        self._char_api.End()

    def _preprocessing(self, images):
        return [vyto_thresh(im) for im in images]

    def chars2strings(self, images):
        images = self._preprocessing(images)
        return [img2string(im, api=self._char_api) for im in images]
    
    def numbers2strings(self, images):
        images = self._preprocessing(images)
        return [img2string(im, api=self._line_api) for im in images]

    def horizontal_numbers2strings(self, images):
        images = self._preprocessing(images)
        return [img2string(im, api=self._hori_api) for im in images]

    def vertical_numbers2strings(self, images):
        images = self._preprocessing(images)
        return [img2string(im, api=self._vert_api) for im in images]

class TesseractLegacySingle(TesseractSingle):
    lstm = False

class TesseractLSTMSingle(TesseractSingle):
    lstm = True