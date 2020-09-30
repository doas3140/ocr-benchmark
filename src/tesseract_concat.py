
from .benchmark import BenchmarkInterface
from .thresholds import togray, threshold, vyto_thresh
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
from PIL import Image
import numpy as np
import imutils
import time

def init_api(**kwargs):
    api = PyTessBaseAPI(**kwargs)
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("tessedit_do_invert", "0")
    return api

def img2string(img, api):
    api.SetImage(Image.fromarray(img))
    return api.GetUTF8Text()

class TesseractConcat(BenchmarkInterface):
    lstm = False
    def __init__(self):
        super().__init__()
        oem = OEM.LSTM_ONLY if self.lstm else OEM.TESSERACT_ONLY
        lang = 'eng'
        self._line_api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)
        self._char_api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)

    def __del__(self):
        self._line_api.End()
        self._char_api.End()

    def _preprocessing(self, images):
        return [vyto_thresh(im) for im in images]

    def chars2strings(self, images):
        images = self._preprocessing(images)
        images = [imutils.resize(im, height=60) for im in images]
        return [s for s in img2string(np.concatenate(images, axis=1), api=self._char_api)]
    
    def numbers2strings(self, images):
        images = self._preprocessing(images)
        images = [imutils.resize(im, width=300) for im in images]
        return img2string(np.concatenate(images, axis=0), api=self._line_api) # needs to split more

class TesseractLegacyConcat(TesseractConcat):
    lstm = False

class TesseractLSTMConcat(TesseractConcat):
    lstm = True