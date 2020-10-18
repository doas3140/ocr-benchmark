from .utils.img_utils import morph, togray, threshold, contour_area, find_all_contours
from .utils.ocr import OCRBatch
import cv2


class BR_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.1,.8,.1))
        return threshold(gray, 60, cv2.THRESH_BINARY)

class BL_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.1,.8,.1))
        return threshold(gray, 100, cv2.THRESH_BINARY)

class TR_OCR(OCRBatch):
    def _preprocessing(self, bgr):
        gray = togray(bgr, rgb=(.2,.65,.15))
        gray = self.clahe.apply(gray)
        thresh = threshold(gray, 90, cv2.THRESH_BINARY)
        return thresh
    def _postprocess(self, y_pred):
        y_pred = y_pred.strip()
        for x_str, y_str in [(' ', ''),('!','1'),('|','1'),('I','1')]:
            y_pred = y_pred.replace(x_str, y_str)
        # if self.lstm and len(y_pred) == 10:
        #     y_pred = filter53(y_pred)
        return y_pred

def filter53(y_pred):
    prev_s = ''
    for i,s in enumerate(y_pred):
        if s == '3' and prev_s == '5':
            y_pred = y_pred[:i-1] + y_pred[i:]
            break
        prev_s = s
    return y_pred

class VE_OCR(OCRBatch):
    def _preprocessing(self, thresh): # (treshold w/ unet)
        # gray = togray(bgr, (0,1,0))
        # gray = cv2.medianBlur(gray, 5)
        # thresh = threshold(gray, 100, cv2.THRESH_BINARY)
        # cnts = find_all_contours(255-thresh)
        # cnts = list(filter(lambda c: contour_area(c) < 50, cnts))
        # thresh = cv2.drawContours(thresh, cnts, -1, 255, -1)
        return thresh
    def _postprocess(self, y_pred):
        y_pred = y_pred.strip()
        for s in [' ','\n',"'",'"','.','‘','-','«','(',')','’']:
            y_pred = y_pred.replace(s,'')
        for x_str, y_str in [('/', '7'), ('?', '7'), ('&', '8'), ('€', '6')]:
            y_pred = y_pred.replace(x_str, y_str)
        return y_pred