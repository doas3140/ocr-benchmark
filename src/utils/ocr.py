

from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
import cv2
import PIL


def init_api(**kwargs):
    api = PyTessBaseAPI(**kwargs)
    # outputbase digits
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("classify_bln_numeric_mode", "1")
    api.SetVariable("tessedit_do_invert", "0")
    # api.SetVariable("dpi", "300")
    return api

class OCR:
    def __init__(self, vert=False, lstm=False, lang='eng', **kwargs):
        self.lstm = lstm
        self.api = init_api(psm=PSM.SINGLE_BLOCK_VERT_TEXT if vert else PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY if lstm else OEM.TESSERACT_ONLY, lang=lang)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    def _preprocessing(self, image):
        return image
    def _postprocess(self, y_pred):
        y_pred = y_pred.strip().replace(' ', '')
        return y_pred
    def predict(self, image):
        thresh = self._preprocessing(image)
        self.api.SetImage(PIL.Image.fromarray(thresh))
        return self._postprocess(self.api.GetUTF8Text())

class OCRBatch(OCR):
    def predict(self, image):
        thresh = self._preprocessing(image)
        self.api.SetImage(PIL.Image.fromarray(thresh))
        return [self._postprocess(y_pred) for y_pred in self.api.GetUTF8Text().split('\n') if len(y_pred)>5]
