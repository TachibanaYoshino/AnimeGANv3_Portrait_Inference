import argparse
import cv2
import numpy as np
import os, subprocess
from tqdm import tqdm
from core import pipeline
from core.utils import get_image_file_list
from config import model_args
style_list = list(model_args.v3_face_style.keys())

def argsparse():
    parser = argparse.ArgumentParser(description='AnimeGANv3')
    parser.add_argument('-i', '--input',type=str,default='/data/', help='Input image or image folder or mp4 video file. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='./USA', help='Output folder. Default: USA')
    parser.add_argument('-s', '--style', type=str, choices=[x for x in style_list], help='style for face transformation')
    parser.add_argument('-b', '--background', action='store_true', help='Whether to convert the background')
    parser.add_argument('-c', '--save_croped', action='store_true', help='save the croped face')
    parser.add_argument('-t', '--IfConcat', type=str, default="None", choices=["None", "Horizontal", "Vertical"], help='Whether to splice the original video with the converted video')
    parser.add_argument('-p', '--use_parse', type=bool, default=True, help='Segmentation face. Default: True')
    args = parser.parse_args()
    return args


def Conversion(img, v3_models, background, use_parse=True):
    if isinstance(v3_models, str) and v3_models in style_list:
        models_init = pipeline.Models()
        model_face, model_bg = models_init.get_v3_face_model(v3_models)
        face_det, face_seg = models_init.get_face_tools()
    else:
        model_face, model_bg, face_det, face_seg  = v3_models
    img = pipeline.read_image(img)
    cropped_faces, restored_faces, restored_img = pipeline.transform_image(img, [model_face, model_bg, face_det, face_seg], background, use_parse)
    if isinstance(v3_models, str) and v3_models in style_list:
        models_init.v3_face_model_destroy()
        models_init.face_tools_destroy()
    return cropped_faces, restored_faces, restored_img   # bgr


def image_enforce(input, output, v3_models, background, save_croped, use_parse):
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
        cropped_faces, restored_faces, restored_img = Conversion(input_img, v3_models, background, use_parse)
        # save faces
        if save_croped:
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                pipeline.imwrite(cropped_face, save_crop_path)
                # save restored face
                save_restore_path = os.path.join(output, 'restored_faces', f'{basename}_{idx:02d}.png')
                pipeline.imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                pipeline.imwrite(cmp_img, os.path.join(output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            save_restore_path = os.path.join(output, 'imgs', img_name)
            pipeline.imwrite(restored_img, save_restore_path)
    print(f'Results are in the [{output}] folder.')


def video_enforce(input, output, v3_models, background, IfConcat, use_parse):
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
        cropped_faces, restored_faces, restored_img = Conversion(frame, v3_models, background, use_parse)
        restored_img = pipeline.icv_resize(restored_img,  (width, height))
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
    style = args.style
    background = args.background
    save_croped = args.save_croped
    IfConcat = args.IfConcat
    use_parse = args.use_parse

    models_init = pipeline.Models()
    model_face, model_bg = models_init.get_v3_face_model(style)
    face_det, face_seg = models_init.get_face_tools()
    # ------------------------ input & output ------------------------
    if not (input.endswith('.mp4') or input.endswith('.MP4')):
        image_enforce(input, output, [model_face, model_bg, face_det, face_seg], background, save_croped, use_parse)
    else:
        video_enforce(input, output, [model_face, model_bg, face_det, face_seg], background, IfConcat, use_parse)
    models_init.v3_face_model_destroy()
    models_init.face_tools_destroy()


