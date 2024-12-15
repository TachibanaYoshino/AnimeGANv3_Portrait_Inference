import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from retina_cfg import  cfg_mnet, cfg_re50, PriorBox, py_cpu_nms, decode, decode_landm
from rknn.api import RKNN

resize = 1
confidence_threshold = 0.8
top_k = 50
nms_threshold = 0.4
keep_top_k = 20
cfg = cfg_re50  # cfg_mnet


def convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL,IMG_PATH ):

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[104., 117., 123.]], std_values=[[1, 1, 1]],
                target_platform='rk3588', dynamic_input = [[[1, 3, IMG_SIZE, IMG_SIZE]]], quant_img_RGB2BGR=[True])
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
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # resize
    height, width = img.shape[0], img.shape[1]
    max_edge = max(width, height)
    scale_factor = IMG_SIZE / max_edge if max_edge > IMG_SIZE else 1.
    height, width = int(round(height * scale_factor)), int(round(width * scale_factor))
    img = cv2.resize(img, (width, height))
    temp = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=img.dtype)
    temp[:height, :width, :] = img
    img = temp
    resize = 1 / scale_factor
    # ----
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # if scale
    # img -= np.array([104., 117., 123.])  # BGR
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format=['nchw'])
    print('done')

    # post process
    loc, conf, landms = outputs
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.forward()

    boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale * resize
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
    scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                       img.shape[3], img.shape[2]])
    landms = landms * scale1 * resize
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)

    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    # dets = dets[:keep_top_k, :4] # get rid of score
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    # dets = np.concatenate((dets, landms), axis=1)
    box_order = np.argsort((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[::-1]
    dets = dets[box_order, :]
    landms = landms[box_order, :]
    # landms = np.reshape(landms, (landms.shape[0], 5, 2))
    # if 0 in dets.shape:
    #     return None, None
    # return dets, landms
    res = np.concatenate((dets, landms), axis=1)

    rknn.release()

    return res

if __name__ == '__main__':

    IMG_PATH = './007_a.jpg'
    DATASET = './dataset.txt'
    QUANTIZE_ON = True
    IMG_SIZE = 840

    ONNX_MODEL = './model_core/Resnet50_Final.onnx'
    RKNN_MODEL = ONNX_MODEL.replace('onnx', 'rknn')

    preds = convert(ONNX_MODEL, QUANTIZE_ON, DATASET, RKNN_MODEL, IMG_PATH)

    img = cv2.imread(IMG_PATH)

    # print(preds.shape)
    num = 0 if preds is None else len(preds)
    print(f'face num: {num} ')
    boxes, points = preds[:, :4], preds[:, 5:15]
    points = points.reshape((points.shape[0], -1, 2))
    # print(boxes.shape)
    # Draw boxes and save faces
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        w = box[2] - box[0]
        h = box[3] - box[1]
        box = [int(x) for x in box]
        draw.rectangle((box), width=2, outline=(255, 0, 0))
        for i, p in enumerate(point):
            x, y = int(p[0]), int(p[1])
            draw.rectangle((x - 5, y - 5, x + 5, y + 5), width=2, outline=(255, 255 - 51 * i, 51 * i))

    cv2.imwrite('face.jpg', np.array(img_draw)[:, :, ::-1])