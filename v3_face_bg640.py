import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
from rknnlite.api import RKNNLite

try:
    from .config import  model_args

except:
    from config import  model_args




class V3_FACE_bg():
    def __init__(self, rknn_model=model_args.USA_model_path):

        self.rknn_lite = RKNNLite(verbose=False,)
        ret1 = self.rknn_lite.load_rknn(rknn_model)
        ret2 = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret1 != 0 or ret2 != 0:
            print(f'Load {rknn_model} model failed')
            exit(ret1)

    def destroy(self):
        self.rknn_lite.release()

    def icv_resize(sefl,mat, size):
        img = Image.fromarray(mat)
        img = img.resize(size, Image.ANTIALIAS)
        return np.array(img)

    def get_scale_shape(self, img, limit=640):
        height, width = img.shape[0], img.shape[1]
        max_edge = max(width, height)
        scale_factor = limit / max_edge if max_edge > limit else 1.
        height = int(round(height * scale_factor))
        width = int(round(width * scale_factor))
        img = self.icv_resize(img, (width, height))
        temp = np.zeros((640, 640, 3), dtype=img.dtype)
        temp[:height, :width, :] = img
        img = temp
        return img, (width, height)

    def v3_preprocess(self, img):
        ori_shape = (img.shape[1], img.shape[0])
        img, scale_WH = self.get_scale_shape(img)
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)
        return img, ori_shape, scale_WH

    def v3_post_processing(self, pred, ori_shape, scale_WH):
        img = pred[0]
        # img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = img[:, :, ::-1]  # RGB -> BGR
        img = (img + 1) * 127.5
        img = img.clip(0, 255).astype(np.uint8)
        img = img[:scale_WH[1], :scale_WH[0], :]
        img = self.icv_resize(img, ori_shape)
        return img

    def Stylize_face(self, img):
        x, ori_shape, scale_WH = self.v3_preprocess(img)
        # Inference
        outputs = self.rknn_lite.inference(inputs=[x], data_format=['nchw'])[0]
        restored_face = self.v3_post_processing(outputs, ori_shape, scale_WH)
        return restored_face


if __name__ == '__main__':
    v3_face = V3_FACE_bg(model_args.USA_model_path_bg)
    img_path = r"./data/120.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    out = v3_face.Stylize_face(img)
    cv2.imwrite('out.jpg', np.hstack([cv2.imread(img_path), out]))
    v3_face.destroy()

