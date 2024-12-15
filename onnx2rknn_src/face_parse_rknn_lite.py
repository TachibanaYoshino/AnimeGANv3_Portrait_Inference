import os
import traceback
import time
import sys
import numpy as np
import cv2

from rknnlite.api import RKNNLite


def convert(rknn_model, IMG_PATH ):
    host_name = 'RK3588'

    # Create RKNN object
    rknn_lite = RKNNLite()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
    if host_name in ['RK3576', 'RK3588']:
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (512, 512))
    img = img[:, :, ::-1]  # BGR -> RGB
    # img = img.astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img], data_format=['nchw'])
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
    cv2.imwrite('parse.jpg', np.hstack([cv2.imread(IMG_PATH), cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    # cv2.imwrite('parse.jpg', mask.astype(np.uint8))

    rknn_lite.release()

if __name__ == '__main__':

    IMG_PATH = './007_a.jpg'

    rknn_model = './model_core/parsing_parsenet_sim.rknn'

    convert(rknn_model, IMG_PATH )