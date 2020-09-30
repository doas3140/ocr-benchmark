import timeit
import numpy as np
import cv2
from pathlib import Path
import random
from src.benchmark import BenchmarkInterface
from src.thresholds import threshold
import inspect
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from time import time
from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr


def print_3visaconcat_benchmark():
    benchmark_name = 'ALL CONCAT (in minutes)'
    print('\n################## START {} ({}) #########################'.format(benchmark_name))
    bboxes = np.array([
        [ 945,   70,  400,   75],
        [ 250, 1110,  350,  100],
        [ 980, 1110,  350,  100],
        [ 465,  210,   70,  480],
        [ 945, 1485,  400,   75],
        [ 250, 2535,  350,  100],
        [ 980, 2535,  350,  100],
        [ 465, 1629,   70,  480],
        [ 945, 2905,  400,   75],
        [ 250, 3965,  350,  100],
        [ 980, 3965,  350,  100],
        [ 465, 3061,   70,  480]])
    
    concat_h = np.sum(bboxes,axis=0)[3]
    concat_w = np.max(bboxes,axis=0)[2]
    
    def predict_one_image():
        with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY, lang='eng') as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetVariable("tessedit_do_invert", "0")
            concat_i = np.empty((concat_h,concat_w,3), np.uint8)
            visa = cv2.imread('./data/0.png')
            y = 0
            concat_i.fill(255)
            for bb in bboxes:
                concat_i[y:y+bb[3],0:0+bb[2]]=visa[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
                y+=bb[3]
        
            concat_filtered = threshold(concat_i, 80, cv2.THRESH_BINARY)
            concat_g_img = Image.fromarray(concat_filtered)
            api.SetImage(concat_g_img)
            _ = api.GetUTF8Text()
    t0 = time()
    for _ in range(100): predict_one_image()
    print(f'TOTAL TIME: {(time()-t0)*100}')
    print('################## END {} ({}) #########################'.format(benchmark_name, kwargs['name']))