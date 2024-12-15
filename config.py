import os, easydict
import numpy as np
opj = os.path.join
pwd = os.path.abspath(os.path.dirname(__file__))

face_model_path = "model_core"
model_args=easydict.EasyDict({
    "Retina_face_model_path": f'{opj(pwd, face_model_path, "Resnet50_Final.rknn")}',
    "parsing_parsenet_model_path": f'{opj(pwd, face_model_path, "parsing_parsenet_sim.rknn")}',
})

model_512_path =r"."
model_args.USA_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_USA.rknn"
model_args.USA2_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_USA2.rknn"
model_args.Kpop_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_large_Kpop.rknn"
model_args.Arcane_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Arcane.rknn"
model_args.Cute_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Cute.rknn"
model_args.Comic_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Comic.rknn"
model_args.Pixar_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_Pixar.rknn"
model_args.Sketch0_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Sketch-0.rknn"
model_args.Disney2_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_large_Disney2.0.rknn"
model_args.Trump2_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_large_Trump2.0.rknn"
model_args.Nordic_m1_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Nordic_myth1.rknn"
model_args.Nordic_m2_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_Nordic_myth2.rknn"
model_args.bit8_model_path = rf"{model_512_path}/rknn_files_512/AnimeGANv3_light_8bit.rknn"

model_640_path =r"."
model_args.USA_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_USA.rknn"
model_args.USA2_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_USA2.rknn"
model_args.Kpop_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_large_Kpop.rknn"
model_args.Arcane_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Arcane.rknn"
model_args.Cute_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Cute.rknn"
model_args.Comic_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Comic.rknn"
model_args.Pixar_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_Pixar.rknn"
model_args.Sketch0_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Sketch-0.rknn"
model_args.Disney2_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_large_Disney2.0.rknn"
model_args.Trump2_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_large_Trump2.0.rknn"
model_args.Nordic_m1_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Nordic_myth1.rknn"
model_args.Nordic_m2_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_Nordic_myth2.rknn"
model_args.bit8_model_path_bg = rf"{model_640_path}/rknn_files_640/AnimeGANv3_light_8bit.rknn"