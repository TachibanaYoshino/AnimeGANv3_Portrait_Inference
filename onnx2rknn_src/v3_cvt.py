import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN


def convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL,IMG_PATH ):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]],
                target_platform='rk3588', dynamic_input = [[[1,IMG_SIZE,IMG_SIZE,3]]])
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img2 = np.expand_dims(img, 0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img2], data_format=['nhwc'])
    print('done')

    # post process
    print(outputs)

    rknn.release()

if __name__ == '__main__':

    IMG_PATH = './007_a.jpg'
    DATASET = './dataset.txt'
    QUANTIZE_ON = True
    IMG_SIZE = 512

    ONNX_MODEL = './onnx_files/AnimeGANv3_large_Trump2.0.onnx'
    RKNN_MODEL = ONNX_MODEL.replace('onnx', 'rknn')

    convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL, IMG_PATH)