import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
from rknnlite.api import RKNNLite

try:
    from .config import  model_args

except:
    from config import  model_args




class PareseFACE():
    def __init__(self, rknn_model=model_args.Retina_face_model_path):
        self.resize = 1

        self.rknn_lite = RKNNLite(verbose=False)
        ret1 = self.rknn_lite.load_rknn(rknn_model)
        ret2 = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret1 != 0 or ret2 != 0:
            print(f'Load {rknn_model} model failed')
            exit(ret1)

    def destroy(self):
        self.rknn_lite.release()


    def parse_face(self, img):
        img = cv2.resize(img, (512, 512))
        img = img[:, :, ::-1]  # BGR -> RGB
        # img = img.astype(np.float32) / 127.5 - 1.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        # Inference
        outputs = self.rknn_lite.inference(inputs=[img], data_format=['nchw'])

        # post process
        out = outputs[0]
        out = np.argmax(out, axis=1).squeeze()

        return out


if __name__ == '__main__':
    pares = PareseFACE()
    img_path = r"../data/a1/4.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    # det face landmark
    out = pares.parse_face(img)
    # print(preds.shape)
    mask = np.zeros(out.shape)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        mask[out == idx] = color
    print(np.max(mask), np.min(mask), np.unique(mask))
    cv2.imwrite('parse.jpg', np.hstack([cv2.imread(img_path), cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    pares.destroy()

