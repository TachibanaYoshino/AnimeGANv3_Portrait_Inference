import cv2, os, argparse
import numpy as np
try:
    import face_det, face_parse
    from config import model_args
    from v3_face_512 import V3_FACE
    from v3_face_bg640 import V3_FACE_bg
except:
    from . import face_det, face_parse
    from .config import model_args
    from .v3_face_512 import V3_FACE
    from .v3_face_bg640 import V3_FACE_bg

from PIL import Image

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

 # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
def get_face_landmarks_5(input_img, eye_dist_threshold=None):
    face_detector = face_det.RetinaFACE(model_args.Retina_face_model_path)
    bboxes = face_detector.detect_face(input_img)
    face_detector.destroy()
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
            parse_detector = face_parse.PareseFACE(model_args.parsing_parsenet_model_path)
            out = parse_detector.parse_face(restored_face)
            parse_detector.destroy()

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


def Conversion(img, style, background):
    img = read_image(img)
    det_faces_box, all_landmarks_5 = get_face_landmarks_5(img)
    cropped_faces, affine_matrices = align_warp_face(img, all_landmarks_5, face_size=512)
    face_num = len(cropped_faces)
    if style == "USA":
        model = V3_FACE(model_args.USA_model_path)
        model_bg = V3_FACE_bg(model_args.USA_model_path_bg)
    elif style == "USA2":
        model = V3_FACE(model_args.USA2_model_path)
        model_bg =  V3_FACE_bg(model_args.USA2_model_path_bg)
    elif style == "Comic":
        model = V3_FACE(model_args.Comic_model_path)
        model_bg = V3_FACE_bg(model_args.Comic_model_path_bg)
    elif style == "Cute":
        model = V3_FACE(model_args.Cute_model_path)
        model_bg = V3_FACE_bg(model_args.Comic_model_path_bg)
    elif style == "8bit":
        model = V3_FACE(model_args.bit8_model_path)
        model_bg = V3_FACE_bg(model_args.bit8_model_path_bg)
    elif style == "Arcane":
        model = V3_FACE(model_args.Arcane_model_path)
        model_bg = V3_FACE_bg(model_args.Arcane_model_path_bg)
    elif style == "Pixar":
        model = V3_FACE(model_args.Pixar_model_path)
        model_bg = V3_FACE_bg(model_args.Pixar_model_path_bg)
    elif style == "Kpop":
        model = V3_FACE(model_args.Kpop_model_path)
        model_bg = V3_FACE_bg(model_args.Kpop_model_path_bg)
    elif style == "Sketch-0": #
        # model = V3_FACE(model_args.Sketch0_model_path)
        model_bg = V3_FACE_bg(model_args.Sketch0_model_path_bg)
        img = model_bg.Stylize_face(img)
        model_bg.destroy()
        return img
    elif style == "Nordic_myth1":
        model = V3_FACE(model_args.Nordic_m1_model_path)
        model_bg = V3_FACE_bg(model_args.Nordic_m1_model_path_bg)
    elif style == "Nordic_myth2":
        model =V3_FACE(model_args.Nordic_m2_model_path)
        model_bg =  V3_FACE_bg(model_args.Nordic_m2_model_path_bg)
    elif style == "Trump2.0":
        model = V3_FACE(model_args.Trump2_model_path)
        model_bg =  V3_FACE_bg(model_args.Trump2_model_path_bg)
    else:
        model= V3_FACE(model_args.Disney2_model_path)
        model_bg= V3_FACE_bg(model_args.Disney2_model_path_bg)
    # face restoration
    restored_faces = []
    for cropped_face in cropped_faces: # 512*512 aligned faces
        restored_face = model.Stylize_face(cropped_face)
        restored_faces.append(restored_face)
    if background:
        img = model_bg.Stylize_face(img)
    inverse_affine_matrices = get_inverse_affine(affine_matrices, upscale_factor=1)
    restored_img = paste_faces_to_image(img, restored_faces, inverse_affine_matrices, upscale_factor=1, use_parse=True)
    # return cropped_faces, restored_faces, restored_img
    model_bg.destroy()
    model.destroy()
    return restored_img


def argsparse():
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--input',type=str,default='/data/', help='Input image or image folder or mp4 video file. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='./USA', help='Output folder. Default: USA')
    parser.add_argument('-s', '--style', type=str, choices=[
        "USA", "USA2", "Comic", "Cute", "8bit", "Arcane", "Pixar", "Kpop", "Sketch-0", "Nordic_myth1", "Nordic_myth2",
        "Trump2.0", "Disney2.0"
    ])
    parser.add_argument('-b', '--background', action='store_true', help='Whether to convert the background')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparse()
    input, output = args.input, args.output
    os.makedirs(args.output, exist_ok=True)
    select_style = args.style
    background = 'yes' if args.background else 'no'
    formats = ['.jpg', '.jpeg', '.png']
    formats = formats + [x.upper() for x in formats]
    files = [x for x in os.listdir(input) if os.path.splitext(x)[-1] in formats]
    for i, x in enumerate(files):
        out = Conversion(os.path.join(input, x), select_style, background)
        cv2.imwrite(os.path.join(output, x), out)