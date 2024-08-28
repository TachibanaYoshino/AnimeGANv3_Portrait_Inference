import cv2,os
import numpy as np
try:
    import face_det
    from config_core import model_args, ort_sess_options
except:
    from . import face_det
    from .config_core import model_args, ort_sess_options

import onnxruntime as ort
from PIL import Image


device_name = ort.get_device()
providers=None
if device_name == 'CPU':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider' ]#, 'CPUExecutionProvider']

ort_session = ort.InferenceSession(model_args.parsing_parsenet_model_path, sess_options=ort_sess_options, providers=providers)

to_16s = lambda x: 512 if x < 512 else x - x%16

def get_scale_shape(img, limit=1920):
    height, width = img.shape[0], img.shape[1]
    max_edge = max(width, height)
    scale_factor = limit / max_edge if max_edge > limit else 1.
    height = int(round(height * scale_factor))
    width = int(round(width * scale_factor))
    return ( to_16s(width), to_16s(height) )


def icv_resize(mat, size):
    img = Image.fromarray(mat)
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img)

def read_image(img):
    """img can be image path or cv2 loaded image."""
    # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
    if isinstance(img, str):
        img = cv2.imread(img)
    if np.max(img) > 256:  # 16-bit image
        img = img / 65535 * 255
    if len(img.shape) == 2:  # gray image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img = img[:, :, 0:3]
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')

def face_parse(x):
    pred = ort_session.run(None, {ort_session.get_inputs()[0].name: x})[0]
    return pred

 # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
def get_face_landmarks_5(input_img, eye_dist_threshold=None):
    bboxes = face_det.detect_face(input_img)
    all_landmarks_5 = []
    det_faces = []
    for bbox in bboxes:
        # remove faces with too small eye distance: side faces or too small faces
        eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
        if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
            continue
        landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
        all_landmarks_5.append(landmark)
        det_faces.append(bbox[0:5])
    return det_faces, all_landmarks_5

def get_inverse_affine(affine_matrices, upscale_factor=1):
    """Get inverse affine matrix."""
    inverse_affine_matrices = []
    for idx, affine_matrix in enumerate(affine_matrices):
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= upscale_factor
        inverse_affine_matrices.append(inverse_affine)
    return inverse_affine_matrices

def align_warp_face(input_img, all_landmarks_5, face_size=512):
    """Align and warp faces with face template.
    """
    face_template = np.array([
        [192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
        [201.26117, 371.41043], [313.08905, 371.15118]
    ])
    face_template = face_template * (face_size / 512.0)
    cropped_faces = []
    affine_matrices = []
    for idx, landmark in enumerate(all_landmarks_5):
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
        affine_matrices.append(affine_matrix)
        # warp and crop faces
        border_mode = cv2.BORDER_CONSTANT
        cropped_face = cv2.warpAffine(input_img, affine_matrix, (face_size, face_size),borderMode=border_mode,borderValue=(135, 133, 132))  # gray
        cropped_faces.append(cropped_face)

    return cropped_faces, affine_matrices

def paste_faces_to_image(img, restored_faces, inverse_affine_matrices,upscale_factor=1, use_parse=True, face_size=512):
    h, w, _ = img.shape
    for restored_face, inverse_affine in zip(restored_faces, inverse_affine_matrices):
        # Add an offset to inverse affine matrix, for more precise back alignment
        if upscale_factor > 1:
            extra_offset = 0.5 * upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w, h))

        if use_parse:
            # inference
            face_input = icv_resize(restored_face, (512, 512))
            face_input = preprocess(face_input)
            out = face_parse(face_input)
            out = np.argmax(out, axis=1).squeeze()

            mask = np.zeros(out.shape)
            MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
            for idx, color in enumerate(MASK_COLORMAP):
                mask[out == idx] = color
            #  blur the mask
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            # remove the black borders
            thres = 10
            mask[:thres, :] = 0
            mask[-thres:, :] = 0
            mask[:, :thres] = 0
            mask[:, -thres:] = 0
            mask = mask / 255.

            mask = icv_resize(mask, restored_face.shape[:2])
            mask = cv2.warpAffine(mask, inverse_affine, (w, h), flags=3)
            inv_soft_mask = mask[:, :, None]
            pasted_face = inv_restored

        else:  # use square parse maps
            mask = np.ones((face_size, face_size), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
            # remove the black borders
            inv_mask_erosion = cv2.erode(inv_mask, np.ones((int(2 * upscale_factor), int(2 * upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)  # // 3
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area ** 0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)

            if len(img.shape) == 2:  # upsample_img is gray image
                img = img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]

        if len(img.shape) == 3 and img.shape[2] == 4:  # alpha channel
            alpha = img[:, :, 3:]
            img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * img[:, :, 0:3]
            img = np.concatenate((img, alpha), axis=2)
        else:
            img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * img
    # img = img.astype(np.uint8)
    return img.clip(0,255).astype(np.uint8)

def v3_preprocess(img):
    ori_shape = (img.shape[1], img.shape[0])
    img = icv_resize(img, get_scale_shape(img))
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img, ori_shape

def v3_post_processing(pred, ori_shape):
    img = pred[0]
    # img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR
    img = (img + 1) * 127.5
    img = img.clip(0,255).astype(np.uint8)
    img = icv_resize(img, ori_shape)
    return img

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR
    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)
    return img.clip(0,255)