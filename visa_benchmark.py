
import os
import timeit
import numpy as np
import cv2
from pathlib import Path
import random
import inspect
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import time
from PIL import Image
import pytesseract
import PIL
import time
import pandas as pd
from collections import defaultdict
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
import imutils
from tabulate import tabulate

class Timer:
    def __init__(self):
      self.D = defaultdict(lambda: [])
    def time(self, name):
      self.name = name
      return self
    def __enter__(self):
      self.t0 = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
      self.D[self.name].append(time.time() - self.t0)
    def get_means_df(self):
      d = {f'{k} ({len(v)})':[np.sum(v)] for k,v in self.D.items()}
    #   d.update({f'{k} (total)':[np.sum(v)] for k,v in self.D.items()})
      return pd.DataFrame(d)

def morph(gray, morph_type=cv2.MORPH_TOPHAT, kernel=(3,3)):
  return cv2.morphologyEx(gray, morph_type, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))

def togray(bgr, rgb=(0.2989,0.5870,0.1140)):
  r,g,b = rgb
  return (b*bgr[:,:,0] + g*bgr[:,:,1] + r*bgr[:,:,2]).astype(bgr.dtype)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def threshold(gray, val=0, thresh_type=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
  return cv2.threshold(gray, val, 255, thresh_type)[1]

def init_api(**kwargs):
    api = PyTessBaseAPI(**kwargs)
    # api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("classify_bln_numeric_mode", "1")
    api.SetVariable("tessedit_do_invert", "0")
    return api

class OCR:
    def __init__(self, vert=False, lstm=False, lang='eng', **kwargs):
        self.api = init_api(psm=PSM.SINGLE_BLOCK_VERT_TEXT if vert else PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY if lstm else OEM.TESSERACT_ONLY, lang=lang)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    def _preprocessing(self, image):
        return image
    def _postprocess(self, y_pred):
        return y_pred.strip().replace(' ', '')
    def predict(self, image):
        thresh = self._preprocessing(image)
        self.api.SetImage(PIL.Image.fromarray(thresh))
        return self._postprocess(self.api.GetUTF8Text())

class OCRBatch(OCR):
    def predict(self, image):
        thresh = self._preprocessing(image)
        self.api.SetImage(PIL.Image.fromarray(thresh))
        return [self._postprocess(y_pred) for y_pred in self.api.GetUTF8Text().split('\n') if y_pred not in '']


class BR_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.1,.8,.1))
        return threshold(gray, 70, cv2.THRESH_BINARY)

class BL_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.1,.8,.1))
        return threshold(gray, 120, cv2.THRESH_BINARY)

class TR_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.2,.65,.15))
        gray = self.clahe.apply(gray)
        return threshold(gray, 80, cv2.THRESH_BINARY)
    def _postprocess(self, y_pred):
        y_pred = y_pred.strip()
        for x_str, y_str in [(' ', ''),('!','1')]:
            y_pred = y_pred.replace(x_str, y_str)
        return y_pred

class VE_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        bgr = togray(bgr, (0,1,0))
        bgr = cv2.medianBlur(bgr, 5)
        return threshold(bgr, 80, cv2.THRESH_BINARY)
    def _postprocess(self, y_pred):
        y_pred = y_pred.strip()
        for s in [' ','\n',"'",'"','.','‘','-','«','(',')','’']:
            y_pred = y_pred.replace(s,'')
        for x_str, y_str in [('?', '7'), ('&', '8'), ('€', '6')]:
            y_pred = y_pred.replace(x_str, y_str)
        return y_pred

def find_all_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)
  
def contour_area(contour):
    moments = cv2.moments(contour)
    return moments['m00']

def find_digits(bgr, pad=4, clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))):
    bgr_orig = imutils.resize(bgr, width=60)
    bgr = bgr_orig.copy()
    bgr[bgr[:,:,1] > 100] = 0 # filter yellow
    bgr[bgr[:,:,2] < 80] = 0 # filter black
    bgr = togray(bgr, [int(x)/10 for x in str(901)])
    bgr = clahe.apply(bgr)
    # bgr = cv2.medianBlur(bgr, 5)
    bgr = threshold(bgr, 75, cv2.THRESH_BINARY)
    # bgr = morph(bgr, cv2.MORPH_OPEN, (2,2))
    contours = find_all_contours(bgr)
    contours = list(filter(lambda c: contour_area(c) > 45, contours))
    bboxes = list(map(cv2.boundingRect, contours))
    H,W = bgr_orig.shape[:2]
    digits = []
    for x,y,w,h in sorted(bboxes, key=lambda b: b[1]):
        t,l,b,r = max(0,y-pad), max(0,x-pad), min(H,y+h+pad), min(W,x+w+pad)
        digit = bgr_orig[t:b,l:r]
        digits.append(digit)
    return digits


x,w,h = 240,400,120
bl_bb = [
  [ x, 1120,  w,  h],
  [ x, 2580,  w,  h],
  [ x, 4050,  w,  h]
]

x = 980
br_bb = [
  [ x, 1120,  w,  h],
  [ x, 2580,  w,  h],
  [ x, 4050,  w,  h]
]

x,w,h = 965,410,100
tr_bb = [
  [ x, 50,  w,   h],
  [ x, 1510,  w,   h],
  [ x, 2960,  w,   h]
]

x,w,h = 475,75,480
ve_bb = [
  [ x,  210,   w,  h],
  [ x, 1670,   w,  h],
  [ x, 3130,   w,  h]
]

TIMER = Timer()

class ImageDataset(Dataset):
    def __init__(self, dir_path):
        super().__init__()
        self.image_paths = list(Path(dir_path).iterdir())
        self.start_nr = 2851000
        self.diffs = [0, int(1e4), int(2e4)]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        p = self.image_paths[idx]
        image = cv2.imread(str(p))
        im_nr = int(p.name[:-4])
        nrs = ['00' + str(self.start_nr-im_nr + diff) for diff in self.diffs]
        crops = []
        for name, bbs in zip(['bl','br','tr','ve'], [bl_bb, br_bb, tr_bb, ve_bb]):
            for bb, nr in zip(bbs, nrs):
                crop = image[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
                cv2.normalize(crop, crop, 0, 255, cv2.NORM_MINMAX)
                if name == 've':
                    digits = [imutils.resize(im, height=36) for im in find_digits(crop)]
                    crop = np.concatenate(digits, axis=1)
                crop = imutils.resize(crop, width=500)
                crops.append((name, crop, nr))
        return crops

bl_ocr, br_ocr, tr_ocr, ve_ocr = BL_OCR(), BR_OCR(), TR_OCR(lstm=True), VE_OCR()

dataset = ImageDataset('./visa/')
dl = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=os.cpu_count(), collate_fn=lambda x:x[0])

t_start = time.time()
i = 0
all_crops = []
print('READING IMAGES...')
with TIMER.time(f'imread'):
    for crops in tqdm(dl):
        i += 1
        # if i >20: break
        all_crops.extend(crops)
print()

def name2ocr(name):
    if name == 'bl': return bl_ocr
    if name == 'br': return br_ocr
    if name == 'tr': return tr_ocr
    if name == 've': return ve_ocr
    assert False

bl_crops = list(filter(lambda x: x[0] == 'bl', all_crops))
br_crops = list(filter(lambda x: x[0] == 'br', all_crops))
tr_crops = list(filter(lambda x: x[0] == 'tr', all_crops))
ve_crops = list(filter(lambda x: x[0] == 've', all_crops))

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


for name, ocr, crops in zip(['bl','br','tr','ve'], [bl_ocr, br_ocr, tr_ocr, ve_ocr], [bl_crops, br_crops, tr_crops, ve_crops]):
    images = list(map(lambda x: x[1], crops))
    y_preds = []
    for image_batch in batch_list(images, 64):
        big_image = np.concatenate(image_batch, axis=0)
        with TIMER.time(f'pred-{name}'):
            y_preds.extend(ocr.predict(big_image))
    y_trues = list(map(lambda x: x[2], crops))
    try:
        print(f'{name} accuracy: {np.mean(np.array(y_preds) == np.array(y_trues))}')
    except:
        print(f'ERROR: # of preds: {len(y_preds)} # of trues: {len(y_trues)}')
print(f'TOTAL TIME: {time.time() - t_start}')

print(tabulate(TIMER.get_means_df(), headers='keys', tablefmt='psql'))