
from .benchmark import BenchmarkInterface
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
from PIL import Image
import numpy as np
import imutils

def img2string(img, api):
    api.SetImage(Image.fromarray(img))
    return api.GetUTF8Text()

class TesseractConcat(BenchmarkInterface):
    lstm = False
    def __init__(self):
        super().__init__()
        oem = OEM.LSTM_ONLY if self.lstm else OEM.TESSERACT_ONLY
        self._line_api = PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, oem=oem)
        self._char_api = PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, oem=oem)

    def __del__(self):
        self._line_api.End()
        self._char_api.End()

    def chars2strings(self, images):
        images = [imutils.resize(im, height=60) for im in images]
        return [s for s in img2string(np.concatenate(images, axis=1), api=self._char_api)]
    
    def numbers2strings(self, images):
        images = [imutils.resize(im, width=350) for im in images]
        return img2string(np.concatenate(images, axis=0), api=self._line_api).split('\n')

class TesseractLegacyConcat(TesseractConcat):
    lstm = False

class TesseractLSTMConcat(TesseractConcat):
    lstm = True