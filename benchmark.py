import timeit
import numpy as np
import cv2
from pathlib import Path
import random
from src.benchmark import BenchmarkInterface
from src.tesseract_single import TesseractLegacySingle, TesseractLSTMSingle
from src.tesseract_concat import TesseractLegacyConcat, TesseractLSTMConcat
import inspect

def t(func, times=1):
    return np.mean(timeit.repeat(func, repeat=times, number=1))

def gen_batch(arr, num_elems):
    return [random.choice(arr) for _ in range(num_elems)]

def t_list(func, arr, num_batches=1, times=1):
    # creates batches out of arr and returns how long does it take on average to process *ONE* item
    return t(lambda: func(gen_batch(digits, num_batches)), times) / num_batches

digits = [cv2.imread(str(p)) for p in Path('./data').iterdir() if 'digit' in str(p)]
numbers = [cv2.imread('./data/0_0.png'), cv2.imread('./data/0_1.png')]

def print_benchmark_3k(**kwargs):
    benchmark_name = '30K (in minutes)'
    print('\n################## START {} ({}) #########################'.format(benchmark_name, kwargs['name']))
    # chars
    f, batches, mult = kwargs['chars2strings'], [1,8,64], 9*3*10000/60
    times = [t_list(f, numbers, i, times=100)*mult for i in batches]
    for j, i in zip(times, batches): print(f'# one char \t ({i}): \t {j}')
    best_char = np.min(times)
    # numbers
    f, batches, mult = kwargs['numbers2strings'], [1,8,64], 3*3*10000/60
    times = [t_list(f, numbers, i, times=100)*mult for i in batches]
    for j, i in zip(times, batches): print(f'# number \t ({i}): \t {j}')
    best_number = np.min(times)
    print(f'TOTAL: {best_char + best_number}')
    print('################## END {} ({}) #########################'.format(benchmark_name, kwargs['name']))

def obj2dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__'): # and not inspect.ismethod(value):
            pr[name] = value
    return pr

# print_benchmark_3k(**obj2dict(BenchmarkInterface()))
print_benchmark_3k(**obj2dict(TesseractLegacySingle()))
print_benchmark_3k(**obj2dict(TesseractLSTMSingle()))
print_benchmark_3k(**obj2dict(TesseractLegacyConcat()))
print_benchmark_3k(**obj2dict(TesseractLSTMConcat()))
# print_benchmark_3k(**obj2dict(BenchmarkInterface()))