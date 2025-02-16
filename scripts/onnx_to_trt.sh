
onnx_file=$1
save_file=$2

#onnx_file=./onnxs/AnimeGANv3_Disney2.0.onnx
#save_file=./AnimeGANv3_Disney2.0.trt


trtexec --onnx=$onnx_file  \
        --saveEngine=$save_file \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --minShapes=AnimeGANv3_input:0:1x256x256x3 \
        --optShapes=AnimeGANv3_input:0:1x512x512x3 \
        --maxShapes=AnimeGANv3_input:0:16x1024x1024x3