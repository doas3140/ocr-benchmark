import timeit
import numpy as np
import cv2
from pathlib import Path
import random
from concat_benchmark import print_3visaconcat_benchmark
from src.benchmark import BenchmarkInterface
from src.thresholds import threshold
from src.tesseract_single import TesseractLegacySingle, TesseractLSTMSingle
from src.tesseract_concat import TesseractLegacyConcat, TesseractLSTMConcat
from src.tesseract_multicore import TesseractLegacyMultiCore, TesseractLSTMMultiCore
import inspect
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from time import time
# import ray
# ray.init()

import os
os.environ['OMP_THREAD_LIMIT'] = '8'

def t(func, times=1):
    return np.mean(timeit.repeat(func, repeat=times, number=1))

def gen_batch(arr, num_elems):
    return [random.choice(arr) for _ in range(num_elems)]

def t_list(func, arr, num_batches=1, times=1):
    # creates batches out of arr and returns how long does it take on average to process *ONE* item
    return t(lambda: func(gen_batch(arr, num_batches)), times) / num_batches

digits = [cv2.imread(str(p)) for p in Path('./data').iterdir() if 'digit' in str(p)]
numbers = [cv2.imread('./data/0_0.png'), cv2.imread('./data/0_1.png')]

def print_speed_benchmark_3k(**kwargs):
    benchmark_name = '30K (in minutes)'
    print('\n################## START {} ({}) #########################'.format(benchmark_name, kwargs['name']))
    # chars
    f, batches, mult = kwargs['chars2strings'], [1,8,64,128], 9*3*10000/60
    times = [t_list(f, digits, i, times=10)*mult for i in batches]
    for j, i in zip(times, batches): print(f'# one char \t ({i}):   \t {j}')
    best_char = np.min(times)
    # numbers
    f, batches, mult = kwargs['numbers2strings'], [1,8,64,128], 3*3*10000/60
    times = [t_list(f, numbers, i, times=10)*mult for i in batches]
    for j, i in zip(times, batches): print(f'# number \t ({i}):   \t {j}')
    best_number = np.min(times)
    print(f'TOTAL: {best_char + best_number}')
    print('################## END {} ({}) #########################'.format(benchmark_name, kwargs['name']))

ocr_benchmark_data = {
    'vert': [(str(p), p.name[:-4]) for p in Path('./ocr/vert/').iterdir()],
    'tr': [(str(p), p.name[:-4]) for p in Path('./ocr/tr/').iterdir()],
    'br': [(str(p), p.name[:-4]) for p in Path('./ocr/br/').iterdir()],
    'bl': [(str(p), p.name[:-4]) for p in Path('./ocr/bl/').iterdir()]
}

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
            

def print_acc_benchmark(**kwargs):
    benchmark_name = '300 visa (acc in %)'
    print('\n################## START {} ({}) #########################'.format(benchmark_name, kwargs['name']))
    times = []
    for name, data in ocr_benchmark_data.items():
        all_trues, all_preds = [], []
        batches = batch_list(data, 1)
        
        f_time = 0
        for batch in tqdm(batches):
            images = [cv2.imread(im_path) for im_path, y_true in batch]
            f = kwargs['vertical_numbers2strings'] if 'vert' in name else kwargs['horizontal_numbers2strings']
            t_start = time()
            y_preds = f(images)
            f_time += time() - t_start
            y_trues = [y_true for im_path, y_true in batch]
            all_trues.extend(y_trues)
            all_preds.extend(y_preds)
        print(f_time, len(all_trues), len(all_preds))
        t_delta = f_time / len(all_trues)
        print(f'{name}|', end='')
        for fname, f in zip(['accuracy'], [accuracy_score]):
            print(f'{fname}:\t {f(all_trues, all_preds)}|', end='')
        times.append(t_delta * 10000 * 3 / 60)
        print(f'time(/30k): {times[-1]}')
    print(f'TOTAL TIME: {np.sum(times)}')
    print('################## END {} ({}) #########################'.format(benchmark_name, kwargs['name']))

def obj2dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__'): # and not inspect.ismethod(value):
            pr[name] = value
    return pr

print_3visaconcat_benchmark()

# print_speed_benchmark_3k_multicore(**obj2dict(TesseractLSTMNaive()))
# print_speed_benchmark_3k(**obj2dict(BenchmarkInterface()))
# print_speed_benchmark_3k(**obj2dict(TesseractLegacySingle()))
# print_speed_benchmark_3k(**obj2dict(TesseractLSTMSingle()))
# print_speed_benchmark_3k_single(**obj2dict(TesseractLSTMMultiCore()))
# print_speed_benchmark_3k(**obj2dict(TesseractLegacyConcat()))
# print_speed_benchmark_3k(**obj2dict(TesseractLSTMConcat()))

# print_speed_benchmark_3k(**obj2dict(TesseractLegacySingle()))