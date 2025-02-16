import os, easydict
import numpy as np
opj = os.path.join
pwd = os.path.abspath(os.path.dirname(__file__))

cuda_device=0

model_path = "models"
model_args=easydict.EasyDict({
    "Retina_face_model_path": f'{opj(pwd, model_path, "Resnet50_Final.trt")}',  # int8
    "parsing_parsenet_model_path": f'{opj(pwd, model_path, "parsing_parsenet.trt")}',  # fp16
    "v3_face_style": {},
})

# face models of AnimeGANv3
model_args.v3_face_style["USA"] = rf"{model_path}/v3_face/AnimeGANv3_light_USA.trt"
model_args.v3_face_style["USA2"] = rf"{model_path}/v3_face/AnimeGANv3_light_USA2.trt"
model_args.v3_face_style["Kpop"] = rf"{model_path}/v3_face/AnimeGANv3_large_Kpop.trt"
model_args.v3_face_style["Arcane"] = rf"{model_path}/v3_face/AnimeGANv3_light_Arcane.trt"
model_args.v3_face_style["Cute"] = rf"{model_path}/v3_face/AnimeGANv3_light_Cute.trt"
model_args.v3_face_style["Comic"] = rf"{model_path}/v3_face/AnimeGANv3_light_Comic.trt"
model_args.v3_face_style["Pixar"] = rf"{model_path}/v3_face/AnimeGANv3_Pixar.trt"
model_args.v3_face_style["Sketch-0"] = rf"{model_path}/v3_face/AnimeGANv3_light_Sketch-0.trt"
model_args.v3_face_style["Disney2.0"] = rf"{model_path}/v3_face/AnimeGANv3_large_Disney2.0.trt"
model_args.v3_face_style["Trump2.0"] = rf"{model_path}/v3_face/AnimeGANv3_large_Trump2.0.trt"
model_args.v3_face_style["Nordic_myth1"] = rf"{model_path}/v3_face/AnimeGANv3_light_Nordic_myth1.trt"
model_args.v3_face_style["Nordic_myth2"] = rf"{model_path}/v3_face/AnimeGANv3_light_Nordic_myth2.trt"
model_args.v3_face_style["8bit"] = rf"{model_path}/v3_face/AnimeGANv3_light_8bit.trt"
model_args.v3_face_style["PortraitSketch"] = rf"{model_path}/v3_face/AnimeGANv3_PortraitSketch_25.trt"
model_args.face_shape = [512,512]
model_args.bg_shape = [768,768]
