import numpy as np
import argparse
import imutils
import cv2
from pathlib import Path
from functools import partial
from imutils import contours
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import IPython
import random
import PIL
import copy
from sklearn.metrics import recall_score, precision_score
from ray import tune
from tabulate import tabulate
from src.utils.img_utils import morph, togray, threshold, normalize, clahe, bgr2cmyk

NUM_SAMPLES = 100

### MODEL

def model(bgr, config): # bgr[h,w,3] -> gray[h,w]
    if config['normalize'] is not None:
        bgr = normalize(bgr, config['normalize'])
    cmyk = bgr2cmyk(bgr)
    gray = 255-cmyk[:,:,1]
    if config['clahe']:
        gray = clahe(gray)
    if config['blur'] is not None:
        gray = cv2.medianBlur(gray, config['blur'])
    thresh = threshold(gray, config['thresh'], cv2.THRESH_BINARY)
    return thresh

config = {
    'clahe': (tune.choice([True, False]), True),
    'normalize': (tune.choice([cv2.NORM_MINMAX, cv2.NORM_L2, None]), cv2.NORM_L2),
    'blur': (tune.choice([None,3,5,7]), 5),
    'thresh': (tune.randint(10, 250), 100),
}

### EVAL LOOP

image_paths = [str(p) for p in Path('data/images').iterdir()]
label_paths = [str(p) for p in Path('data/labels').iterdir()]
numbers = [p.name[:-4] for p in Path('data/images').iterdir()]

def eval_func(config):
    try:
        p, r, l = 0, 0, 0 # precision, recall, loss
        for im_p,y_p,nr in zip(image_paths,label_paths,numbers):
            im,y_true = cv2.imread(f'../../{im_p}'), cv2.imread(f'../../{y_p}') # NOTE: must need ../../
            h,w = y_true.shape[:2]
            y_pred = cv2.resize(model(im, config), (w,h))
            y_true = y_true[:,:,0] # (all 3 channels should be same)
            y_pred, y_true = map(lambda x: (255-x).flatten().astype(bool).astype(int), [y_pred, y_true])
            l += np.mean((y_true-y_pred)**2)
            r += recall_score(y_true, y_pred)
            p += precision_score(y_true, y_pred)
    except:
        p, r, l = 9999, 9999, 9999
    scores = {'p':p, 'r':r, 'l':l}
    tune.report(**scores)

### TUNE RAY

analysis = tune.run(
    eval_func,
    num_samples=NUM_SAMPLES,
    config={k:v[0] for k,v in config.items()},
    progress_reporter=tune.CLIReporter(max_progress_rows=50),
    local_dir='.',
    resume=False
)

df = analysis.results_df
df.sort_values(by=['l'], inplace=True)

df = df.drop(columns=['done', 'time_this_iter_s', 'time_since_restore', 'pid', 'timesteps_total', 'episodes_total', 'training_iteration', 'experiment_id', 'date', 'timestamp', 'hostname', 'node_ip', 'timesteps_since_restore', 'iterations_since_restore', 'experiment_tag'])

df.to_csv('hyperparameter_results.csv')

print(tabulate(df, headers='keys', tablefmt='psql'))