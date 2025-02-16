
# AnimeGANv3 Portrait Inference by rknn


## Introduction
1. AnimeGANv3's portrait style model focuses more on the stylized transformation of the face area. Such as Kpop, USA, Disney, Trump, Nordic_myth2 and Arcane.
2. This repo mainly deals with situations where multiple faces appear in images.
3. The program uses face detection to detect each face in the image and then transforms it separately, and then uses the homography matrix and head segmentation to restore the transformed faces to the original image. 
4. Since portrait style models are mainly trained using close-up face images, it may not be applicable when the face area in the image is particularly small or blurry. 

## Usage  

### 1. Install Dependencies  
   ```bash
   pip install -r requirements.txt
   ```

### 2. Model conversion  

Use the script files in **onnx2rknn_src** to convert the onnx models of retianface, parsenet, and AnimeGANv3 to rknn models. 
- Modify the script parameters according to the existing chip version and then perform the conversion.

### 3. Inference using rknn.  
 ```bash
    python convert.py -i data -o out -s Kpop --background
  ```

##### 🔸 Parameter Description
- -i , The location of the input image or the directory where the images are located
- -s , Select the AnimeGANv3 style you want to convert  
- -o , The directory location where the conversion results are saved
- --background , Whether to perform stylization on the background area other than the face

 

### 4. Using the webUI 
- Use the webUI.py script on the edge device to start the browser UI interface, which can be used online. The screenshots are as follows: 

![webUI](./screenshot.jpg)


## Note
1. The computing power of edge devices is limited, so when converting the full image (background), the original image is reduced to a fixed scale of 640*640 for background conversion.  
2. The **TensorRT**-based model conversion and inference [repo](https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/tree/tensorRT). 

