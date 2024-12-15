import os
import traceback
import time
from utils import check_folder
import numpy as np
import cv2
from rknn.api import RKNN


def convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL,IMG_PATH ,out_path):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588', dynamic_input = [[[1,IMG_SIZE, IMG_SIZE,3]]])
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
    img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img2 = img1.astype(np.float32) / 127.5 - 1.0
    img2 = np.expand_dims(img2, axis=0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img2], data_format=['nchw'])
    print('done')

    # post process
    print(type(outputs), len(outputs))
    print(outputs[0].shape)
    out = outputs[0][0]
    out = (out + 1) * 127.5
    out = out.clip(0, 255).astype(np.uint8)
    save_dir = check_folder(f'./res_{IMG_SIZE})')
    cv2.imwrite(f'{save_dir}/res_{out_path}.jpg', np.hstack([img1, out])[:, :, ::-1] )

    rknn.release()

if __name__ == '__main__':

    IMG_PATH = './007_a.jpg'
    DATASET = './dataset.txt'
    QUANTIZE_ON = False
    IMG_SIZE = 640 # background
    # IMG_SIZE = 512 # face

    path = './onnx_files'
    onnxfs = os.listdir(path)
    for x in onnxfs:

        ONNX_MODEL = os.path.join(path, x)
        RKNN_MODEL = ONNX_MODEL.replace('onnx', 'rknn')
        os.makedirs(os.path.dirname(RKNN_MODEL), exist_ok=True)
        out_path = os.path.basename(x).replace('.onnx', '')
        convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL, IMG_PATH, out_path)