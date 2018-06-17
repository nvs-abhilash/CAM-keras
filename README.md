# Class Activation Mapping visulaization

Class Activation Mapping is a way to enables the convolutional neural network to have remarkable localization ability despite being trained on image-level labels. I use it to visualize what my model is looking in the images.

The theory is described here:  [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)

The script `cam_keras.py` is the Keras implementation of Class Activation Mapping. It is based on [this](https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py) script in pytorch.

## Installing Dependencies
This project depends on latest version of 

* `tensorflow`
* `pillow`
* `opencv-python`
* `numpy`
* `requests`

Do `pip install <missing_package>` to install whatever you don't have. Using `conda` environment is recommended. 

## Running cam_keras
To get the output you have to run:

```
python3 cam_keras.py
```
**Note**: 
1. To change the model, change the `model_id` in the script.
2. The output image is stored in `CAM.jpg`.
3. To change the input image, modify the `IMG_URL` in the script.


## Example result

### Input image
![Input image](https://github.com/nvs-abhilash/CAM-keras/blob/master/assets/test.jpg "Input Image")

### Output image
![CAM image](https://github.com/nvs-abhilash/CAM-keras/blob/master/assets/CAM.jpg "CAM Image")
