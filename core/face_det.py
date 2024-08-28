import cv2
import os, time
import onnxruntime
from PIL import Image, ImageDraw
import numpy as np
try:
    from config_core import model_args, ort_sess_options
    from retinaface_ import cfg_mnet, cfg_re50
    from retinaface_.prior_box import PriorBox
    from retinaface_.py_cpu_nms import py_cpu_nms
    from retinaface_.box_utils import decode, decode_landm
except:
    from .config_core import model_args, ort_sess_options
    from .retinaface_ import cfg_mnet, cfg_re50
    from .retinaface_.prior_box import PriorBox
    from .retinaface_.py_cpu_nms import py_cpu_nms
    from .retinaface_.box_utils import decode, decode_landm


device_name = onnxruntime.get_device()
providers=None
if device_name == 'CPU':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider']

ort_session = onnxruntime.InferenceSession(model_args.Retina_face_model_path, sess_options=ort_sess_options,providers=providers)  # mobilenet
cfg = cfg_re50 # cfg_mnet



def detect_face(img, resize=1, confidence_threshold=0.8, top_k=50, nms_threshold=0.4, keep_top_k=20):
    # resize
    height, width = img.shape[0], img.shape[1]
    max_edge = max(width, height)
    scale_factor = 840 / max_edge if max_edge > 840 else 1.
    height, width  = int(round(height * scale_factor)),   int(round(width * scale_factor))
    img = cv2.resize(img, (width, height))
    temp = np.zeros((840,840,3),dtype=img.dtype)
    temp[:height, :width,:] = img
    img = temp
    resize = 1/scale_factor
    # ----
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # if scale
    img -= np.array([104., 117., 123.]) # BGR
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    loc, conf, landms = ort_outs
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
    return np.concatenate((dets, landms), axis=1)






if __name__ == '__main__':
    img_path = r"../data/a1/4.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    # det face landmark
    preds = detect_face(img)
    # print(preds.shape)
    num = 0 if preds is None else len(preds)
    print(f'face num: {num} ')
    boxes, points = preds[:,:4], preds[:,5:15]
    points = points.reshape((points.shape[0], -1, 2))
    # print(boxes.shape)
    # Draw boxes and save faces
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        w = box[2] - box[0]
        h = box[3] - box[1]
        box = [int(x) for x in box ]
        draw.rectangle((box), width=2, outline=(255, 0, 0))
        for i, p in enumerate(point):
            x, y = int(p[0]), int(p[1])
            draw.rectangle((x- 5, y-5, x+ 5, y+5), width=2, outline=(255, 255 - 51 * i, 51 * i))
    cv2.imshow('s', np.array(img_draw)[:,:,::-1])
    cv2.waitKey(0)

