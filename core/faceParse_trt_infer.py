import cv2
import os, time
from PIL import Image, ImageDraw
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class PareseFACE_TRTWrapper():
    def __init__(self, engine_path, batch=1, device=0):
        self.cfx = cuda.Device(device).make_context()
        self.batch = batch  # batch_size
        self.engine_path = engine_path  # batch_size

        self.image_size = (512, 512)
        self.IN_SHSPE = [self.batch, 3, self.image_size[0], self.image_size[1]]
        self.OUT_SHSPE = [self.batch, 19, self.image_size[0], self.image_size[1]]

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.h_input = cuda.pagelocked_empty(trt.volume(self.IN_SHSPE), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)

        self.h_output0 = cuda.pagelocked_empty(trt.volume(self.IN_SHSPE), dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(trt.volume(self.OUT_SHSPE), dtype=np.float32)
        self.d_output0 = cuda.mem_alloc(self.h_output0.nbytes)
        self.d_output1 = cuda.mem_alloc(self.h_output1.nbytes)
        self.stream = cuda.Stream()

    def preprocess(self, images):
        if not isinstance(images, list):
            images = [images]
        out = []
        for img in images:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = img[:, :, ::-1]  # BGR -> RGB
            img = img.astype(np.float32) / 127.5 - 1.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.expand_dims(img, axis=0)
            out.append(img)
        return np.concatenate(out, axis=0)

    def postprocess_image(self, outs):
        outputs = []
        for pred in outs:
            out = np.argmax(pred, axis=0).squeeze()
            outputs.append(out)
        return np.array(outputs)

    def destory(self):
        self.cfx.pop()
        pass
    def parse_face(self, images):
        self.cfx.push()
        input_tensors = self.preprocess(images)
        # print(input_tensors.shape)
        np.copyto(self.h_input, np.array(input_tensors).ravel())
        # Inference
        with self.engine.create_execution_context() as context:
            context.set_input_shape("input", input_tensors.shape)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output0), int(self.d_output1)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output0, self.d_output0, self.stream)
            cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream)
            self.stream.synchronize()

            # post process
            out = self.h_output1.reshape(self.OUT_SHSPE)
            self.cfx.pop()
            return self.postprocess_image(out)


if __name__ == '__main__':
    batch = 16
    pares = PareseFACE_TRTWrapper('../models/parsing_parsenet.trt', batch)
    img_path = r"../data/Lucy.jpg"
    print(os.path.splitext(img_path))
    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512,512))
    # det face landmark
    out = pares.parse_face([img]*batch)
    # out = pares.parse_face(img)
    print(out.shape)
    out = out[-1]
    mask = np.zeros(out.shape)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        mask[out == idx] = color
    print(np.max(mask), np.min(mask), np.unique(mask))
    cv2.imwrite('a.jpg', np.hstack([img, cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    pares.destory()

