from flask import Flask, request, Response, send_file
import jsonpickle
import numpy as np
import os
from tqdm import tqdm
from time import time
from waitress import serve
import tempfile
import cv2

# from fastai import *
# from fastai.vision import *

app = Flask(__name__)

### TEST ROUTES ###

def decode_image(img_bytes):
    img_raw = np.frombuffer(img_bytes, np.uint8) # <class 'bytes'> -> np.arr
    img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
    img = img[ : , : , [2,1,0] ] # BGR -> RGB
    return img

def send_image(img, config):
    r = request.args.get('resize')
    with tempfile.NamedTemporaryFile() as f:
        img = img[ : , : , [2,1,0] ] # RGB -> BGR
        if r is not None: img = cv2.resize(img, dsize=None, fx=float(r), fy=float(r))
        cv2.imwrite(f'{f.name}.jpg', img)
        return send_file(f'{f.name}.jpg', mimetype='image/gif') # as_attachment=True


''' @input:
@input: Body = {'image': File})
@output: {'message': 'image received...'}
'''
@app.route('/api/test', methods=['POST'])
def test():
    img = decode_image(img_bytes=request.files['image'].read())
    response = {'message': 'image received. img.shape: {}x{}'.format(img.shape[1], img.shape[0])}
    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")


''' 
@input: (
    Body = {'image': File},
    Params = {}
)
@output: jpg image
'''
@app.route('/api/test_image', methods=['POST'])
def test_image():
    config = dict(request.args)
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        return send_image(img, config=config)
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


PRODUCTION = True
if PRODUCTION:
    serve(app, host="0.0.0.0", port=5000)
else:
    app.run(host="0.0.0.0", port=5000)