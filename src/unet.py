
from .utils.img_utils import morph, togray, threshold, contour_area, find_all_contours
from .utils.other import batch_list
from fastai.vision.all import *
import imutils
import cv2
import numpy as np
from tqdm import tqdm

DEVICE = 'cuda'
learn = load_learner('./models/model')
learn.model.eval()
learn.model.to(DEVICE)

def treshold_vertical_crops(crops, bs=8, thresh=0.999, device=DEVICE):
    # returns horizontal tresholded crops
    ims_out = []
    for ims in tqdm(batch_list(crops, bs)):
        ims = list(map(lambda i: i[1], ims))
        ims = tensor([cv2.resize(im, (64,128*4)) for im in ims]).to(device)
        ims = ims[ :, :, :, [2,1,0] ] # BGR -> RGB
        ims = tensor(ims).permute(0,3,1,2).float() / 255.
        ys = learn.model(ims).cpu().detach().numpy()
        mask = np.zeros_like(ys, dtype=np.uint8)
        mask[ys > 1-thresh] = 255
        mask = list(map(lambda y: y[0], mask))
        ims_out.extend(mask)
    ims_out = [imutils.resize(im, width=500) for im in ims_out]
    ims_out = [vert2hori(remove_small_blobs_vert(im)) for im in ims_out]
    ims_out = [imutils.resize(im, width=500) for im in ims_out]
    crops_out = []
    for thresh, (name,crop,nr) in zip(ims_out,crops):
        crops_out.append((name,thresh,nr))
    return crops_out

def remove_small_blobs_vert(thresh): # only works for vertical text
    cnts = find_all_contours(255-thresh)
    bboxes = list(map(cv2.boundingRect, cnts))
    x_mean = np.median(list(map(lambda x: x[0], bboxes)))
    def filter_func(cnt):
        a = contour_area(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        return a < 13000 or (h < 30) or (w < 10) # or abs(x - x_mean) > 20
    cnts = list(filter(filter_func, cnts))
    thresh = cv2.drawContours(thresh, cnts, -1, 255, -1)
    return thresh

def vert2hori(thresh, pad=20):
    contours = find_all_contours(255-thresh)
    digits = []
    H,W = thresh.shape[:2]
    for x,y,w,h in sorted(map(cv2.boundingRect, contours), key=lambda b: b[1]):
        t,l,b,r = max(0,y-pad), max(0,x-pad), min(H,y+h+pad), min(W,x+w+pad)
        digits.append(thresh[t:b,l:r])
    digits = [imutils.resize(im, height=60) for im in digits]
    return np.concatenate(digits, axis=1)