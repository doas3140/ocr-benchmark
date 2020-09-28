
from .benchmark import BenchmarkInterface
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
from PIL import Image

def img2string(img, api):
    api.SetImage(Image.fromarray(img))
    return api.GetUTF8Text()

class TesseractSingle(BenchmarkInterface):
    lstm = False
    def __init__(self):
        super().__init__()
        oem = OEM.LSTM_ONLY if self.lstm else OEM.TESSERACT_ONLY
        self._line_api = PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, oem=oem)
        self._char_api = PyTessBaseAPI(psm=PSM.SINGLE_CHAR, oem=oem)

    def __del__(self):
        self._line_api.End()
        self._char_api.End()

    def chars2strings(self, images):
        return [img2string(im, api=self._char_api) for im in images]
    
    def numbers2strings(self, images):
        return [img2string(im, api=self._line_api) for im in images]

class TesseractLegacySingle(TesseractSingle):
    lstm = False

class TesseractLSTMSingle(TesseractSingle):
    lstm = True