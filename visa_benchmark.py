
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
import imutils
from tabulate import tabulate
import matplotlib.pyplot as plt
from fastai.vision.all import *
from src.utils.img_utils import morph, togray, threshold, contour_area, find_all_contours
from src.utils.other import batch_list, Timer
from src.ocr import BL_OCR, BR_OCR, TR_OCR, VE_OCR
label_func = lambda x:x # load_model BUG
from src.unet import treshold_vertical_crops

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

x,w,h = 975,400,100
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
    def __init__(self, dir_path, slice=None):
        super().__init__()
        if slice is None:
            self.image_paths = list(sorted(Path(dir_path).iterdir()))
        else:
            self.image_paths = list(sorted(Path(dir_path).iterdir()))[slice]
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
                if name == 've':
                    pass
                else:
                    crop = imutils.resize(crop, width=500)
                cv2.normalize(crop, crop, 0, 255, cv2.NORM_MINMAX)
                crops.append([name, crop, nr])
        assert len(crops) == 12
        return crops

bl_ocr, br_ocr, tr_ocr, ve_ocr = BL_OCR(), BR_OCR(), TR_OCR(lstm=True, lang='custom300_tr'), VE_OCR()

all_crops = []
dataset = ImageDataset('./visa/')
num_examples = 500
print('READING IMAGES...')
for i in tqdm(range(0, len(dataset), num_examples)):
    dataset = ImageDataset('./visa/', slice=slice(i,i+num_examples))
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=lambda x:x[0], drop_last=False)

    t_start = time.time()
    i = 0
    with TIMER.time(f'imread'):
        for crops in tqdm(dl):
            i += 1
            all_crops.extend(crops)

def pytorch_delistify(x): # BUG
    name,crop,nr = x
    if type(name) is tuple:
        name, crop, nr = name[0], np.array(crop)[0], nr[0]
    return (name,crop,nr)
all_crops = list(map(pytorch_delistify, all_crops))
    
print('NUM CROPS:', len(all_crops))

bl_crops = list(filter(lambda x: x[0] == 'bl', all_crops))
br_crops = list(filter(lambda x: x[0] == 'br', all_crops))
tr_crops = list(filter(lambda x: x[0] == 'tr', all_crops))
ve_crops = list(filter(lambda x: x[0] == 've', all_crops))

print('TRESHOLDING WITH UNET...')
with TIMER.time(f'nn'):
    ve_crops = treshold_vertical_crops(ve_crops)


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

def predict_in_batch(ocr, images, y_trues=[], name=''):
    y_preds = []
    for i, image_batch in enumerate(tqdm(batch_list(images, 3*40))):
        # big_image = np.concatenate(image_batch, axis=1 if name == 've' else 0)
        big_image = np.concatenate(image_batch, axis=0)
        with TIMER.time(f'pred-{name}'):
            y_pred = ocr.predict(big_image)
            if SAVE_CROPS:
                cv2.imwrite(f'big_images/{i}.tiff', ocr._preprocessing(big_image))
            # y_preds.extend(list(reversed(y_pred)) if name == 've' else y_pred)
            y_preds.extend(y_pred)
    return y_preds

def predict_in_solo(ocr, images, y_trues=[], name=''):
    y_preds = []
    for i,image in enumerate(tqdm(images)):
        with TIMER.time(f'pred-{name}'):
            y_pred = ocr.predict(image)
            y_preds.extend(y_pred)
            if SAVE_CROPS and y_pred[0] != y_trues[i]:
                cv2.imwrite(f'big_images/{y_pred[0]}_{y_trues[i]}.tiff', ocr._preprocessing(image))
    return y_preds

PRINT_BAD = True
save_crop_dir = './crops'
SAVE_CROPS = True

print('CHAR RECOGNITION...')
for name, ocr, crops in zip(['bl','br','tr','ve'], [bl_ocr, br_ocr, tr_ocr, ve_ocr], [bl_crops, br_crops, tr_crops, ve_crops]):
    if name in []: continue
    images = list(map(lambda x: x[1], crops))
    y_trues = list(map(lambda x: x[2], crops))
    pred_func = predict_in_solo if name in ['ve'] else predict_in_batch
    y_preds = pred_func(ocr, images, y_trues, name)
    if PRINT_BAD:
        # bads = [x for x in zip(images, y_preds, y_trues) if x[1] != x[2]]
        # fig = plt.figure()
        for i, (im,yp,yt) in enumerate(zip(images, y_preds, y_trues)):
            if yp != yt:
                cv2.imwrite(f'bads/{yp}_{yt}_{name}.png', im)
                print(f'PRED: {yp} TRUE: {yt}')
    try:
        print(f'{name} accuracy: {np.mean(np.array(y_preds) == np.array(y_trues))}')
    except:
        print(f'ERROR: # of preds: {len(y_preds)} # of trues: {len(y_trues)}')
print(f'TOTAL TIME: {time.time() - t_start}')

print(tabulate(TIMER.get_means_df(), headers='keys', tablefmt='psql'))
