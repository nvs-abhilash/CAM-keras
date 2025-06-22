"""
Author: NVS Abhilash

Keras implementation of Class Activation Mappings: http://cnnlocalization.csail.mit.edu/
This script is based on pytorch version of the same: https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

"""

import numpy as np
import cv2
import io
import requests
from PIL import Image

# Using Keras implementation from tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

HEIGHT = 224
WIDTH = 224

# Change this id to use different models (currently ResNet50 and InceptionV3 only)
model_id = 1
if model_id == 1:
    model = keras.applications.ResNet50(include_top=True)
    finalconv_name = 'activation_49'
    preprocess_input = keras.applications.resnet50.preprocess_input
elif model_id == 2:
    model = keras.applications.InceptionV3(include_top=True)
    finalconv_name = 'mixed10'
    preprocess_input = keras.applications.inception_v3.preprocess_input
else:
    raise ValueError('Unsupported model_id: {}'.format(model_id))

# Get the layer of the last conv layer
final_conv = model.get_layer(finalconv_name)

# Get the weights matrix of the last layer
weight_softmax = model.layers[-1].get_weights()[0]

# Function to generate Class Activation Mapping
def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (WIDTH, HEIGHT)

    # Keras default is channels last, hence nc is in last
    bz, h, w, nc = feature_conv.shape

    output_cam = []
    for idx in class_idx:
        cam = np.dot(weight_softmax[:, idx], np.transpose(feature_conv.reshape(h*w, nc)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        output_cam.append(cv2.resize(cam_img, size_upsample))
    
    return output_cam

def main():
    response = requests.get(IMG_URL)
    img_pil = Image.open(io.BytesIO(response.content))
    img_pil.save('test.jpg')

    img_array = img_to_array(load_img('test.jpg', target_size=(HEIGHT, WIDTH)))
    img_array = preprocess_input(img_array)

    probs_extractor = model.predict
    features_conv_extractor = keras.Model(model.input, final_conv.output).predict

    classes = {
        int(key): value
        for (key, value) in requests.get(LABELS_URL).json().items()
    }

    probs = probs_extractor(np.expand_dims(img_array, 0))[0]
    features_blob = features_conv_extractor(np.expand_dims(img_array, 0))[0]

    idx = np.argsort(probs)
    probs = np.sort(probs)

    for i in range(-1, -6, -1):
        print('{:.3f} -> {}'.format(probs[0, i], classes[idx[0, i]]))

    CAMs = returnCAM(features_blob, weight_softmax, [idx[0, -1]])

    print('output CAM.jpg for top1 prediction: {}'.format(classes[idx[0, -1]]))
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(
        cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET
    )

    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)


if __name__ == '__main__':
    main()
