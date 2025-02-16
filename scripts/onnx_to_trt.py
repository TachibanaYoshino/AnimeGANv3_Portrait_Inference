import os, subprocess
from tqdm import tqdm


def run(v3_portrait_onnx_dir, v3_portrait_trt_dir):
    onnx_path =v3_portrait_onnx_dir
    trt_path = v3_portrait_trt_dir
    os.makedirs(trt_path, exist_ok=True)
    onnxs = [x for x in os.listdir(onnx_path) if x.endswith('.onnx')]
    for o in tqdm(onnxs):
        cmd = ['sh', 'onnx_to_trt.sh', f'{os.path.join(onnx_path, o)}', f"{os.path.join(trt_path, o.replace('.onnx', '.trt'))}"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("ERROR INFO: ")
            print(result.stderr)
        else:
            print(result.stdout)


if __name__ == '__main__':
    v3_portrait_onnx_dir = r'./models'
    v3_portrait_trt_dir ='./trt_models'
    run(v3_portrait_onnx_dir, v3_portrait_trt_dir)