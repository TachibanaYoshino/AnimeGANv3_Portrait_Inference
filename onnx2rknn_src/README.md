
# AnimeGANv3 Model Conversion for rknn


## Introduction
1. Corresponding to the inference code and model of the main branch. The inference framework is changed from onnx to rknn, and the inference hardware is changed to the NPU chip of the edge device.
2. rknn_toolkit2 currently does not support quantization with NHWC input format for tensorflow and onnx. AnimeGANv3 model converted to f16.


## Script Description  

- v3_cvt_fp16.py , Convert the AnimeGANv3 model to the fp16 rknn model. For the foreground conversion of the face, use a fixed input of 512*512. For the conversion of the background image, use a fixed input of 640*640.
- v3_cvt.py ,  ~~Quantization of AnimeGANv3 models are not supported.~~  
- retinaface_cvt.py , Quantization conversion for retinaface face detection model.
- retinaface_rknn_lite.py , Inference test of retinaface's rknn model.
- face_parse_cvt.py , Quantization transformation for the ParseNet face segmentation model.
- face_parse_rknn_lite.py , Inference test of ParseNet's rknn model.
- imgs/ , A directory containing about 100 face images, used for quantization of the model conversion process.

 

