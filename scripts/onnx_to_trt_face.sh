
#onnx_file=$1
#save_file=$2

onnx_file=./parsing_parsenet.onnx
save_file=./parsing_parsenet.trt
trtexec --onnx=$onnx_file  \
        --saveEngine=$save_file \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --minShapes=input:1x3x512x512 \
        --optShapes=input:1x3x512x512 \
        --maxShapes=input:16x3x512x512


#onnx_file=./Resnet50_Final.onnx
#save_file=./Resnet50_Final.trt
#trtexec --onnx=$onnx_file  \
#        --saveEngine=$save_file \
#        --memPoolSize=workspace:4096 \
#        --fp16 \
#        --minShapes=input0:1x3x640x640 \
#        --optShapes=input0:1x3x840x840 \
#        --maxShapes=input0:16x3x960x960