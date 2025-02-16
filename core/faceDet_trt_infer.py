import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

try:
    from .retina_cfg import  cfg_mnet, cfg_re50, PriorBox, py_cpu_nms, decode, decode_landm

except:
    from retina_cfg import cfg_mnet, cfg_re50, PriorBox, py_cpu_nms, decode, decode_landm

class RetinaFACE_TRTWrapper():

    def __init__(self, engine_path, batch=1, device=0, cfg=cfg_re50, confidence_threshold=0.8, top_k=50, nms_threshold=0.4, keep_top_k=20):
        self.cfx = cuda.Device(device).make_context()
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.cfg = cfg  # cfg_mnet
        self.batch = batch  # batch_size
        self.engine_path = engine_path  # batch_size

        self.image_size = (840, 840)
        self.IN_SHSPE = [self.batch, 3, self.image_size[0], self.image_size[1]]

        self.priorbox = PriorBox(self.cfg, image_size=self.image_size)
        self.prior_data = self.priorbox.forward()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.h_input = cuda.pagelocked_empty(trt.volume(self.IN_SHSPE), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)

        self.h_output0 = cuda.pagelocked_empty(trt.volume([self.batch, 29126, 4]), dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(trt.volume([self.batch, 29126, 10]), dtype=np.float32)
        self.h_output2 = cuda.pagelocked_empty(trt.volume([self.batch, 29126, 2]), dtype=np.float32)
        self.d_output0 = cuda.mem_alloc(self.h_output0.nbytes)
        self.d_output1 = cuda.mem_alloc(self.h_output1.nbytes)
        self.d_output2 = cuda.mem_alloc(self.h_output2.nbytes)
        self.stream = cuda.Stream()
    def proprecess(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        meta = []
        out = []
        for img in imgs:
            height, width = img.shape[0], img.shape[1]
            max_edge = max(width, height)
            scale_factor = 840 / max_edge if max_edge > 840 else 1.
            height, width = int(round(height * scale_factor)), int(round(width * scale_factor))
            img = cv2.resize(img, (width, height))
            temp = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=img.dtype)
            temp[:height, :width, :] = img
            img = temp
            resize = 1 / scale_factor
            # ----
            img = np.float32(img)
            im_height, im_width, _ = img.shape
            scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= np.array([104., 117., 123.])  # # BGR
            img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
            # img = np.ascontiguousarray(img)
            meta.append([scale, resize])
            out.append(img)
        return np.concatenate(out, axis=0), meta

    def postprocess_image(self, outs, meta):
        outputs = []
        locs, confs, landmss = outs
        for i in range(len(meta)):
            loc, conf, landms = locs[i], confs[i], landmss[i]
            scale, resize = meta[i][0], meta[i][1]
            boxes = decode(loc.squeeze(), self.prior_data, self.cfg['variance'])
            boxes = boxes * scale * resize
            scores = conf.squeeze()[:, 1]
            landms = decode_landm(landms.squeeze(), self.prior_data, self.cfg['variance'])
            scale1 = np.array([scale[0], scale[1], scale[0], scale[1],
                               scale[0], scale[1], scale[0], scale[1],
                               scale[0], scale[1]])
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
            res = np.concatenate((dets, landms), axis=1)
            outputs.append(res)
        return outputs

    def destory(self):
        self.cfx.pop()
        pass
    def detect_face(self, imgs):
        self.cfx.push()
        input_tensors, meta = self.proprecess(imgs)
        # print(input_tensors.shape)
        np.copyto(self.h_input, np.array(input_tensors).ravel())

        with self.engine.create_execution_context() as context:
            context.set_input_shape("input0", input_tensors.shape)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output0), int(self.d_output1), int(self.d_output2)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output0, self.d_output0, self.stream)
            cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream)
            cuda.memcpy_dtoh_async(self.h_output2, self.d_output2, self.stream)
            self.stream.synchronize()
            loc, landms, conf = self.h_output0.reshape([self.batch,29126,4]), self.h_output1.reshape([self.batch,29126,10]), self.h_output2.reshape([self.batch,29126,2])
            self.cfx.pop()
            return self.postprocess_image([loc, conf, landms], meta)


if __name__ == '__main__':
    batch = 16
    retinaDET = RetinaFACE_TRTWrapper('../models/Resnet50_Final.trt', batch)
    img_path = r"../data/Alice.png"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    # det face landmark
    preds = retinaDET.detect_face([img]*batch)
    # preds = retinaDET.detect_face(img)
    # print(preds)
    print(len(preds))
    preds = preds[-1]
    print(preds.shape)
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
    cv2.imwrite('a.jpg', np.array(img_draw)[:,:,::-1])
    retinaDET.destory()


