import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os, cv2, time



class v3_TRTWrapper():
    def __init__(self, engine_path, batch=1, in_shape=(512,512), device=0):
        self.cfx = cuda.Device(device).make_context()
        self.engine_path = engine_path
        self.batch = batch
        self.in_shape = [batch] + [x for x in in_shape] + [3] # b,h,w,c
        self.out_shape = self.in_shape

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.h_input = cuda.pagelocked_empty(trt.volume(self.in_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.out_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

    def process_image(self, images, model_name):
        if not isinstance(images, list):
            images = [images]
        out=[]
        shapes =[]
        for img in images:
            h, w = img.shape[:2]
            # resize image to multiple of 8s
            def to_8s(x):
                # If using the tiny model, the multiple should be 16 instead of 8.
                if 'tiny' in os.path.basename(model_name):
                    return 512 if x < 512 else x - x % 16
                else:
                    return 512 if x < 512 else x - x % 8
            img = cv2.resize(img, (to_8s(w), to_8s(h)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            assert img.shape[0] == self.in_shape[1] and img.shape[1] == self.in_shape[2]
            img = np.expand_dims(img, axis=0)
            out.append(img)
            shapes.append([w, h])
        return np.concatenate(out, axis=0), shapes
    def postprocess_image(self, images, size):
        out = []
        for i, img in enumerate(images):
            img = (np.squeeze(img) + 1.) / 2 * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            if size:
                img = cv2.resize(img, size[i])
            out.append(img[:,:,::-1]) # rgb to bgr
        return np.array(out)

    def destory(self):
        self.cfx.pop()
        pass
    def forward(self, mats):
        self.cfx.push()
        input_mat, ori_wh = self.process_image(mats, self.engine_path)
        # print(input_mat.shape)
        np.copyto(self.h_input, np.array(input_mat).ravel())

        with self.engine.create_execution_context() as context:
            context.set_input_shape("AnimeGANv3_input:0", input_mat.shape)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            out = self.h_output.reshape(self.out_shape)
            self.cfx.pop()
            return self.postprocess_image(out, ori_wh)


if __name__ == '__main__':
    model_path = r'../models/v3_face/AnimeGANv3_Disney2.0.trt'
    image_path ='../data/Kobe.png'

    in_shape = (512, 512)
    img0 = cv2.imread(image_path)
    img = cv2.resize(img0, (in_shape[1],in_shape[0]))
    batch = 16
    model = v3_TRTWrapper(model_path, batch, in_shape)
    out = model.forward([img]*batch)
    print(out.shape)
    cv2.imwrite('a.jpg', out[0])
    model.destory()