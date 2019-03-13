from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input

import numpy as np
import cv2


class CAM:
    def __init__(self, model, last_layer):
        """Create a 
        """
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

        self.output_fn = K.function([self.model.input,],
                                    [self.final_conv_layer.output, self.model.output])
        self.softmax_weights = self.model.layers[-1].get_weights()[0]

    def get_activations(self, img_batch):
        return self.output_fn([img_batch])

    def gen_cams(self, featuremaps, softmax_outputs, img_shapes, class_idxs=None):
        cams = np.dot(featuremaps, self.softmax_weights)
        
        if class_idxs is None:
            class_idxs = np.argmax(softmax_outputs, axis=1)
        
        print(np.argsort(softmax_outputs, axis=1)[0, -5:])
        # TODO: Vectorize this
        final_cams = []
        for i, cam in enumerate(cams):
            final_cams += [np.squeeze(cam)[:, :, class_idxs[i]]]
        final_cams = np.array(final_cams)

        print(final_cams.shape)

        # final_cams -= np.min(final_cams, axis=0)
        # final_cams /= (0.1 + np.max(final_cams, axis=0))

        # final_cams = np.uint8(255 * final_cams)

        final_cams = list(final_cams)
        for i, img_shape in enumerate(img_shapes):
            final_cams[i] = cv2.resize(final_cams[i], img_shape)

        return final_cams

    def get_cams_on_batch(self, imgs, img_shapes=None, class_idxs=None):
        """Get CAMs on a batch of images

        If img_shapes is provieded, resizes each image to 

        """
        if img_shapes is None:
            img_shapes = [imgs.shape[1:-1] for i in range(imgs.shape[0])]

        imgs_list = []
        # TODO: Change the image resizing based on the model
        for img in imgs:
            imgs_list += [cv2.resize(img, (224, 224))]
        imgs_arr = np.array(imgs_list)
        imgs_arr = preprocess_input(imgs_arr)

        featuremaps, softmax_outputs = self.get_activations(imgs_arr)
        final_cams = self.gen_cams(featuremaps, softmax_outputs, img_shapes, class_idxs)

        return final_cams
