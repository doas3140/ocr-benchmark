import cv2
import imutils


def morph(gray, morph_type=cv2.MORPH_TOPHAT, kernel=(3,3)):
  return cv2.morphologyEx(gray, morph_type, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))

def togray(bgr, rgb=(0.2989,0.5870,0.1140)):
  r,g,b = rgb
  return (b*bgr[:,:,0] + g*bgr[:,:,1] + r*bgr[:,:,2]).astype(bgr.dtype)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def threshold(gray, val=0, thresh_type=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
  return cv2.threshold(gray, val, 255, thresh_type)[1]

def find_all_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)
  
def contour_area(contour):
    moments = cv2.moments(contour)
    return moments['m00']