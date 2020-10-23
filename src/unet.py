
from .utils.img_utils import morph, togray, threshold, contour_area, find_all_contours
from .utils.other import batch_list
from fastai.vision.all import *
import torch.nn.functional as F
import imutils
import cv2
import numpy as np
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.torch.cuda.is_available() else 'cpu'
print(f'USING {DEVICE} FOR UNET')
nn_img_size = (128*4,64)
out_img_size = (128*4,64)
learn = load_learner(f'./models/model_{nn_img_size[0]}x{nn_img_size[1]}')
learn.model.eval()
learn.model.to(DEVICE)

def pad_image(im, pad=20, pad_val=255):
    h,w = im.shape[:2]
    im_new = np.zeros([h+pad*2,w+pad*2], dtype=im.dtype) + pad_val
    im_new[pad:pad+h,pad:pad+w] = im
    return im_new

def treshold_vertical_crops(crops, bs=8, thresh=0.8, device=DEVICE):
    # returns horizontal tresholded crops
    ims_out = []
    for orig_crops in tqdm(batch_list(crops, bs)):
        orig_ims = list(map(lambda i: i[1], orig_crops))
        ims = tensor([cv2.resize(im, nn_img_size[::-1]) for im in orig_ims]).to(device)
        ims = ims[ :, :, :, [2,1,0] ] # BGR -> RGB
        ims = tensor(ims).permute(0,3,1,2).float() / 255. # [b,3,h,w]
        ys = learn.model(ims)
        ys = F.interpolate(ys, size=out_img_size, mode='bilinear', align_corners=False).cpu().detach().numpy()
        mask = np.zeros_like(ys, dtype=np.uint8)
        mask[ys > 1-thresh] = 255
        mask = list(map(lambda y: y[0], mask))
        for i,(c,m) in enumerate(zip(orig_crops,mask)):
            im, nr = c[1], c[2]
            cv2.imwrite(f'labels/{nr}.png', m)
            cv2.imwrite(f'images/{nr}.png', im)
        ims_out.extend(mask)
    ims_out = [imutils.resize(im, width=500) for im in ims_out]
    ims_out = [vert2hori(remove_small_blobs_vert(im)) for im in ims_out]
    ims_out = [pad_image(im) for im in ims_out]
    ims_out = [imutils.resize(im, width=500) for im in ims_out]
    crops_out = [(name,thresh,nr,p) for thresh, (name,crop,nr,p) in zip(ims_out,crops)]
    return crops_out

def remove_small_blobs_vert(thresh): # only works for vertical text
    cnts = find_all_contours(255-thresh)
    bboxes = list(map(cv2.boundingRect, cnts))
    x_mean = np.median(list(map(lambda x: x[0], bboxes)))
    def filter_func(cnt):
        a = contour_area(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        return a < 12000 or (h < 30) or (w < 10) # or abs(x - x_mean) > 20
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