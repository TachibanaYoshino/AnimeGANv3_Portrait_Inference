
# for images
python onnx_infer.py -i data/a -m ../AnimeGANv3_large_Kpop.onnx -o ./out --background

# for a single video
python onnx_infer.py -i ../1.mp4 -m ../AnimeGANv3_large_Kpop.onnx -o ./out --background

