import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
from rknnlite.api import RKNNLite

try:
    from .config import  model_args

except:
    from config import  model_args



class V3_FACE():
    def __init__(self, rknn_model=model_args.USA_model_path):

        self.rknn_lite = RKNNLite(verbose=False)
        ret1 = self.rknn_lite.load_rknn(rknn_model)
        ret2 = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret1 != 0 or ret2 != 0:
            print(f'Load {rknn_model} model failed')
            exit(ret1)

    def destroy(self):
        self.rknn_lite.release()

    def v3_preprocess(self, img):
        assert img.shape[0] ==512 and img.shape[1] ==512
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)
        return img

    def v3_post_processing(self, pred):
        img = pred[0]
        # img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = img[:, :, ::-1]  # RGB -> BGR
        img = (img + 1) * 127.5
        img = img.clip(0, 255).astype(np.uint8)
        return img

    def Stylize_face(self, img):
        x = self.v3_preprocess(img)
        # Inference
        outputs = self.rknn_lite.inference(inputs=[x], data_format=['nchw'])[0]
        restored_face = self.v3_post_processing(outputs)
        return restored_face


if __name__ == '__main__':
    v3_face = V3_FACE(model_args.USA_model_path)
    img_path = r"./onnx2rknn_src/007_a.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    out = v3_face.Stylize_face(img)
    cv2.imwrite('out.jpg', np.hstack([cv2.imread(img_path), out]))
    v3_face.destroy()

