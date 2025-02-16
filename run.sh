

inputs=$1
out_folder=$2
style=$3

# for images
python convert.py -i ${inputs}  -o ${out_folder} -s ${style} --background

# for a single video
export CUDA_VISIBLE_DEVICES=0 && python convert.py -i ./x.mp4  -o ./out  -s Kpop  --background --IfConcat Horizontal