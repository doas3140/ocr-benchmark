
from .benchmark import BenchmarkInterface
from .thresholds import togray, threshold, vyto_thresh
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
import numpy as np
import cv2
from PIL import Image
import ray
import pytesseract
import multiprocessing
CORES = multiprocessing.cpu_count()

def init_api(**kwargs):
    api = PyTessBaseAPI(**kwargs)
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("tessedit_do_invert", "0")
    return api

@ray.remote
def img2string(image, lstm):
    oem = OEM.LSTM_ONLY if lstm else OEM.TESSERACT_ONLY
    lang = 'eng'
    api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)
    api.SetImage(Image.fromarray(image))
    s = api.GetUTF8Text().strip().replace(' ', '')
    api.End()
    return s
    config = "--psm 6 -c tessedit_do_invert=0 -c tessedit_char_whitelist=0123456789 --oem 1 -l eng --dpi 300"
    return pytesseract.image_to_string(threshold(image, 80, cv2.THRESH_BINARY), config=config)


@ray.remote
def images2strings(images, lstm):
    oem = OEM.LSTM_ONLY if lstm else OEM.TESSERACT_ONLY
    lang = 'eng'
    api = init_api(psm=PSM.SINGLE_BLOCK, oem=oem, lang=lang)
    for image in images:
        api.SetImage(Image.fromarray(image))
        s = api.GetUTF8Text().strip().replace(' ', '')
    api.End()
    return s
    config = "--psm 6 -c tessedit_do_invert=0 -c tessedit_char_whitelist=0123456789 --oem 1 -l eng --dpi 300"
    return [pytesseract.image_to_string(threshold(im, 80, cv2.THRESH_BINARY), config=config) for im in images]

def batch_list(arr, num_batches):
    out = []
    b = []
    for i in arr:
        if len(b) == num_batches:
            out.append(b)
            b = []
        b.append(i)
    if len(b) > 0: out.append(b)
    return out

class TesseractMultiCore(BenchmarkInterface):
    lstm = False
    def __init__(self):
        super().__init__()
    
    def chars2strings(self, images):
        return ray.get([images2strings.remote(im_batch, self.lstm) for im_batch in batch_list(images, CORES)])
        # return ray.get([img2string.remote(im, self.lstm) for im in images])
        # return ray.get([self._char_ray.images2strings.remote(im_batch) for im_batch in batch_list(images, CORES)])
        # return list(self._char_rays.map(lambda a,v: a.images2strings.remote(v), batch_list(images, CORES)))
    
    def numbers2strings(self, images):
        return ray.get([images2strings.remote(im_batch, self.lstm) for im_batch in batch_list(images, CORES)])
        # return ray.get([img2string.remote(im, self.lstm) for im in images])
        # return ray.get([self._line_ray.images2strings.remote(im_batch) for im_batch in batch_list(images, CORES)])
        # return list(self._line_rays.map(lambda a,v: a.images2strings.remote(v), batch_list(images, CORES)))

class TesseractLegacyMultiCore(TesseractMultiCore):
    lstm = False

class TesseractLSTMMultiCore(TesseractMultiCore):
    lstm = True