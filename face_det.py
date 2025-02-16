import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
from rknnlite.api import RKNNLite

try:
    from .retina_cfg import  cfg_mnet, cfg_re50, PriorBox, py_cpu_nms, decode, decode_landm
    from .config import  model_args

except:
    from retina_cfg import cfg_mnet, cfg_re50, PriorBox, py_cpu_nms, decode, decode_landm
    from config import  model_args




class RetinaFACE():
    def __init__(self, rknn_model=model_args.Retina_face_model_path, cfg=cfg_re50, confidence_threshold=0.8, top_k=50, nms_threshold=0.4, keep_top_k=20):
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.cfg = cfg_re50  # cfg_mnet

        self.priorbox = PriorBox(self.cfg, image_size=(840, 840))
        self.prior_data = self.priorbox.forward()

        self.rknn_lite = RKNNLite(verbose=False)
        ret1 = self.rknn_lite.load_rknn(rknn_model)
        ret2 = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret1 != 0 or ret2 != 0:
            print(f'Load {rknn_model} model failed')
            exit(ret1)

    def destroy(self):
        self.rknn_lite.release()


    def detect_face(self, img):
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
        # img -= np.array([104., 117., 123.]) # # BGR
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        outputs = self.rknn_lite.inference(inputs=[img], data_format=['nchw'])
        loc, conf, landms = outputs


        boxes = decode(loc.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * scale * resize
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.squeeze(0), self.prior_data, self.cfg['variance'])
        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = landms * scale1 * resize
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        # dets = dets[:keep_top_k, :4] # get rid of score
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
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
    retinaDET = RetinaFACE()
    img_path = r"../data/120.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    # det face landmark
    preds = retinaDET.detect_face(img)
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
    retinaDET.destroy()

