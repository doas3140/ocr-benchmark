import numpy as np
import cv2
from PIL import Image


def togray(bgr):
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def threshold(gray, val=0, thresh_type=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
  # threshold(gray, 0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # threshold(gray, 50, cv2.THRESH_BINARY_INV)
  return cv2.threshold(gray, val, 255, thresh_type)[1]

def vyto_thresh(bgr):
    b, g, r = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
    g[g>120] = 255
    b[b>120] = 255
    gray = 0*r + 0.5 * g + 0.5 * b
    gray[gray>85] = 255
    return gray.astype(bgr.dtype)