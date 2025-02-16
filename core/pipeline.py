import cv2, os, sys, argparse
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

try:
    import faceDet_trt_infer, faceParse_trt_infer
    from config import model_args, cuda_device
    from v3_trt_infer import v3_TRTWrapper
except:
    from . import faceDet_trt_infer, faceParse_trt_infer
    from .config import model_args, cuda_device
    from .v3_trt_infer import v3_TRTWrapper

from PIL import Image

# Model initialization
class Models():
    def get_face_tools(self):
        self.face_detector = faceDet_trt_infer.RetinaFACE_TRTWrapper(model_args.Retina_face_model_path, device=cuda_device)
        self.parse_detector = faceParse_trt_infer.PareseFACE_TRTWrapper(model_args.parsing_parsenet_model_path, device=cuda_device)
        return  self.face_detector, self.parse_detector

    def face_tools_destroy(self):
        self.face_detector.destory()
        self.parse_detector.destory()

    # V3 Model initialization
    def get_v3_face_model(self, style):
        self.model = v3_TRTWrapper(model_args.v3_face_style[style], in_shape=model_args.face_shape, device=cuda_device)
        self.model_bg = v3_TRTWrapper(model_args.v3_face_style[style], in_shape=model_args.bg_shape, device=cuda_device)
        return self.model, self.model_bg
    def v3_face_model_destroy(self):
        self.model.destory()
        self.model_bg.destory()

def icv_resize(mat, size):
    img = Image.fromarray(mat)
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img)
def bg_precess( img, color=(0, 0, 0)):
    shape = img.shape[:2]  # h,w
    new_shape = model_args.bg_shape  # h,w

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if shape[::-1] != new_unpad:
        img = icv_resize(img, new_unpad)

    top, bottom = int(dh / 2), int(dh - int(dh / 2))
    left, right = int(dw / 2), int(dw - int(dw / 2))
    dw, dh = int(dw / 2), int(dh / 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, new_unpad, dw, dh
def bg_postprecess( out, ori_shape, unpad_shape, dw, dh ):
    unpad = out[dh:dh+unpad_shape[1], dw:dw+unpad_shape[0], :]
    img = icv_resize(unpad, ori_shape)
    return img
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

 # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
def get_face_landmarks_5(input_img, face_detector, eye_dist_threshold=None):
    bboxes = face_detector.detect_face(input_img)[0]
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

def paste_faces_to_image(img, restored_faces, cropped_faces, inverse_affine_matrices, parse_detector, upscale_factor=1, use_parse=True, face_size=512):
    h, w, _ = img.shape
    for restored_face, cropped_face, inverse_affine in zip(restored_faces, cropped_faces, inverse_affine_matrices):
        # Add an offset to inverse affine matrix, for more precise back alignment
        if upscale_factor > 1:
            extra_offset = 0.5 * upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w, h))

        if use_parse:
            # inference
            # out = parse_detector.parse_face(restored_face)[0]
            out = parse_detector.parse_face(cropped_face)[0]

            mask = np.zeros(out.shape)
            MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
            for idx, color in enumerate(MASK_COLORMAP):
                mask[out == idx] = color
            # Prevent the head and the background from having abrupt edges when they merge.
            dist = 50  # Buffer distance from edge
            if np.any(mask[0:dist, :] == 0):
                mask[0:dist, :] = 0
            if np.any(mask[-dist:, :] == 0):
                mask[-dist:, :] = 0
            if np.any(mask[:, 0:dist] == 0):
                mask[:, 0:dist] = 0
            if np.any(mask[:, -dist:] == 0):
                mask[:, -dist:] = 0
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


def transform_image(img, models, background, use_parse=True):
    model_face, model_bg, face_det, face_seg = models
    USE_FACE = True # Whether to capture the face for processing
    if "Sketch-0" in model_face.engine_path or "PortraitSketch" in model_face.engine_path:
        background = True
        USE_FACE = False # Only "Sketch-0" and "PortraitSketch" does not need to capture the face
    bg_img = np.array(img)
    if background:
        ori_shape = [ img.shape[1], img.shape[0] ]
        input, unpad_shape, dw, dh = bg_precess(img)
        output = model_bg.forward(input)[0]
        bg_img = bg_postprecess(output, ori_shape, unpad_shape, dw, dh)
    if USE_FACE:
        det_faces_box, all_landmarks_5 = get_face_landmarks_5(img, face_detector=face_det)
        cropped_faces, affine_matrices = align_warp_face(img, all_landmarks_5, face_size=512)
        # face convertion
        restored_faces = []
        for cropped_face in cropped_faces:  # 512*512 aligned faces
            restored_face = model_face.forward(cropped_face)[0]
            restored_faces.append(restored_face)
        inverse_affine_matrices = get_inverse_affine(affine_matrices, upscale_factor=1)
        restored_img = paste_faces_to_image(bg_img, restored_faces, cropped_faces, inverse_affine_matrices,
                                                     parse_detector=face_seg, upscale_factor=1, use_parse=use_parse)
        return cropped_faces, restored_faces, restored_img
    else:
        return [], [], bg_img