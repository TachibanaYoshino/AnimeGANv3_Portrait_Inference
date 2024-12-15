import os
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
                target_platform='rk3588', dynamic_input = [[[1, 3, 512, 512]]])
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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR -> RGB
    # img = img.astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format=['nchw'])
    print('done')

    # post process
    out = outputs[0]
    print(out.shape)
    out = np.argmax(out, axis=1).squeeze()
    print(out.shape)

    mask = np.zeros(out.shape)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        mask[out == idx] = color
    print(np.max(mask), np.min(mask), np.unique(mask))
    # cv2.imwrite('parse.jpg', np.hstack([cv2.imread(IMG_PATH), cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    cv2.imwrite('parse.jpg', mask.astype(np.uint8))

    rknn.release()

if __name__ == '__main__':

    IMG_PATH = './007_a.jpg'
    DATASET = './dataset.txt'
    QUANTIZE_ON = True
    IMG_SIZE = 512

    ONNX_MODEL = './model_core/parsing_parsenet_sim.onnx'
    RKNN_MODEL = ONNX_MODEL.replace('onnx', 'rknn')

    convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL, IMG_PATH)