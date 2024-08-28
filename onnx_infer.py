import argparse
import cv2
import numpy as np
import os, subprocess
from tqdm import tqdm
import onnxruntime as ort
from core import faceRestoreHelper
from core.config_core import ort_sess_options, get_image_file_list

device_name = ort.get_device()
print(device_name)
providers=None
if device_name == 'CPU':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider'] #, 'CPUExecutionProvider']


def argsparse():
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--input',type=str,default='/data/', help='Input image or image folder or mp4 video file. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='./USA', help='Output folder. Default: USA')
    parser.add_argument('-m', '--model_path', type=str, help='onnx path for style model')
    parser.add_argument('-b', '--background', action='store_true', help='Whether to convert the background')
    parser.add_argument('-c', '--save_croped', action='store_true', help='save the croped face')
    parser.add_argument('-t', '--IfConcat', type=str, default="None", choices=["None", "Horizontal", "Vertical"], help='Whether to splice the original video with the converted video')
    parser.add_argument('-p', '--use_parse', type=bool, default=True, help='Segmentation face. Default: True')
    args = parser.parse_args()
    return args


def Conversion(img, ort_session, background, use_parse):
    img = faceRestoreHelper.read_image(img)
    det_faces_box, all_landmarks_5 = faceRestoreHelper.get_face_landmarks_5(img)
    cropped_faces, affine_matrices = faceRestoreHelper.align_warp_face(img, all_landmarks_5, face_size=512)
    # face restoration
    restored_faces = []
    for cropped_face in cropped_faces: # 512*512 aligned faces
        x, ori_shape  = faceRestoreHelper.v3_preprocess(cropped_face)
        # feedforward
        y = ort_session.run(None, {ort_session.get_inputs()[0].name: x})[0]
        restored_face = faceRestoreHelper.v3_post_processing(y, ori_shape)
        restored_faces.append(restored_face)
    if background:
        x, ori_shape = faceRestoreHelper.v3_preprocess(img)
        pred = ort_session.run(None, {ort_session.get_inputs()[0].name: x})[0]
        img = faceRestoreHelper.v3_post_processing(pred, ori_shape)
    inverse_affine_matrices = faceRestoreHelper.get_inverse_affine(affine_matrices, upscale_factor=1)
    restored_img = faceRestoreHelper.paste_faces_to_image(img, restored_faces, inverse_affine_matrices, upscale_factor=1, use_parse=use_parse)
    return cropped_faces, restored_faces, restored_img


def image_enforce(input, output, ort_session, background, save_croped, use_parse):
    img_list = get_image_file_list(input)
    os.makedirs(output, exist_ok=True)
    # ------------------------ restore ------------------------
    for img_path in tqdm(img_list):
        # read image
        img_name = os.path.basename(img_path)
        # print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path)
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = Conversion(input_img, ort_session, background, use_parse)
        # save faces
        if save_croped:
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                faceRestoreHelper.imwrite(cropped_face, save_crop_path)
                # save restored face
                save_restore_path = os.path.join(output, 'restored_faces', f'{basename}_{idx:02d}.png')
                faceRestoreHelper.imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                faceRestoreHelper.imwrite(cmp_img, os.path.join(output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            save_restore_path = os.path.join(output, 'imgs', img_name)
            faceRestoreHelper.imwrite(restored_img, save_restore_path)
    print(f'Results are in the [{output}] folder.')


def video_enforce(input, output, ort_session, background, IfConcat, use_parse):
    if not (input.endswith('.mp4') or input.endswith('.MP4')):
        raise "input error"
    os.makedirs(output, exist_ok=True)

    vid = cv2.VideoCapture(input)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    pbar = tqdm(total=total )
    pbar.set_description(f"Running: {os.path.basename(input)}")
    ouput_video_path = os.path.join(output,os.path.basename(input))
    ouput_sound_path = os.path.join(output,f'sound.mp3')
    if IfConcat == "Horizontal":
        video_out = cv2.VideoWriter(ouput_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))
    elif IfConcat == "Vertical":
        video_out = cv2.VideoWriter(ouput_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height * 2))
    else:
        video_out = cv2.VideoWriter(ouput_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        cropped_faces, restored_faces, restored_img = Conversion(frame, ort_session, background, use_parse)
        restored_img = faceRestoreHelper.icv_resize(restored_img,  (width, height))
        if IfConcat == "Horizontal":
            restored_img = np.hstack((frame, restored_img))
        elif IfConcat == "Vertical":
            restored_img = np.vstack((frame, restored_img))
        video_out.write(restored_img)
        pbar.update(1)
    pbar.close()
    video_out.release()
    try:
        command = ["ffmpeg", "-loglevel", "error", "-i", input, "-y", ouput_sound_path]
        r = subprocess.check_call(command)  # Get the audio of the input video (MP3)
        ouput_videoSounds_path = ouput_video_path.rsplit('.', 1)[0] + f'_sound.mp4'
        command = ["ffmpeg", "-loglevel", "error", "-i", ouput_sound_path, "-i", ouput_video_path, "-y",
                   ouput_videoSounds_path]
        r = subprocess.check_call(command)  # Merge the output video with the sound to get the final result
    except:
        print("ffmpeg fails to obtain audio, generating silent video.")
    print(f'Results are in the [{output}] folder.')


if __name__ == '__main__':
    args = argsparse()
    input, output = args.input, args.output
    os.makedirs(args.output, exist_ok=True)
    model_path = args.model_path
    background = args.background
    save_croped = args.save_croped
    IfConcat = args.IfConcat
    use_parse = args.use_parse

    pwd = os.path.abspath(os.path.dirname(__file__))
    ort_session = ort.InferenceSession(model_path, sess_options=ort_sess_options, providers=providers)
    # ------------------------ input & output ------------------------
    if not (input.endswith('.mp4') or input.endswith('.MP4')):
        image_enforce(input, output, ort_session, background, save_croped, use_parse)
    else:
        video_enforce(input, output, ort_session, background, IfConcat, use_parse)




