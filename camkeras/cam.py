from tensorflow.python.keras.models import Model
from tensorflow.python.keras import applications
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.resnet50 import preprocess_input

import numpy as np
import cv2


class CAM:
    def __init__(self, model, last_layer):
        if not isinstance(model, Model) and not isinstance(model, str):
            raise ValueError("model should be a Keras model or a str of\
             supported pre-trained models")
        if not isinstance(last_layer, str) and not isinstance(last_layer, int):
            raise ValueError("last_layer must be a str (name of the layer) or\
             int (index of the layer")

        self.supported_models = ['ResNet50', 'InceptionV3']

        if isinstance(model, str):
            if model not in self.supported_models:
                raise ValueError("model when of type str, must be a value\
                                  from the supported model names: {}"
                                 .format(self.supported_models))

            self.model = getattr(applications, model)(include_top=True)
            self.model_name = model
        else:
            self.model = model

        if isinstance(last_layer, str):
            self.final_conv_layer = self.model.get_layer(last_layer)
        else:
            self.final_conv_layer = self.model.layers[last_layer]

        self.output_fn = K.function([self.model.input],
                                    [self.final_conv_layer.output,
                                     self.model.output])

    def get_activations(self, img_batch):
        [self.model_name]
        return self.output_fn([img_batch])

    def gen_cams(self, featuremaps, softmax_preds, img_shapes):
        cams = np.dot(softmax_preds, featuremaps)

        final_cams = []
        for i, img_shape in enumerate(img_shapes):
            final_cams += [cv2.resize(cams[i], img_shape)]

        return final_cams

    def get_cams_on_batch(self, imgs, img_shapes=None):

        if img_shapes is None:
            img_shapes = [imgs.shape[1:-1] for i in range(imgs.shape[0])]

        imgs_list = []
        for img in imgs:
            imgs_list += [cv2.resize(img, (224, 224))]
        imgs_list = np.array(imgs_list)
        imgs_list = preprocess_input(imgs_list)

        featuremaps, softmax_preds = self.get_activations(imgs_list)
        final_cams = self.gen_cams(featuremaps, softmax_preds, img_shapes)

        return final_cams
