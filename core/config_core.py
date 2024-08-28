import os, easydict
import onnxruntime as ort
import numpy as np
opj = os.path.join
pwd = os.path.abspath(os.path.dirname(__file__))

model_args=easydict.EasyDict({
    "Retina_face_model_path": f'{opj(pwd, "model_core", "Resnet50_Final.onnx")}',
    "parsing_parsenet_model_path": f'{opj(pwd, "model_core", "parsing_parsenet_sim.onnx")}',
})

ort_sess_options = ort.SessionOptions()
ort_sess_options.intra_op_num_threads = int(os.environ.get('ort_intra_op_num_threads', 0))


def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'tif', 'tiff'}
    return any([path.lower().endswith(e) for e in img_end])

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists