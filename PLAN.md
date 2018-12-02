# PLAN to move forward

## CAM Visulaization package for Keras - PHASE 1
It has been decided, I would be building a CAM visualization package for keras. 
For that to be made possible I would have to build the following things:

* `camkeras` base library with all the functionality

* `VisCAM` class containing additional tools for visualizing CAMs.
    * `vis_most_correct` in `VisCAM` class.
    * `vis_most_incorrect` in `VisCAM` class.
    * `vis_most_confused` in `VisCAM` class.
    * `heatmap_cams` in `VisCAM` class.

* `CAM` class which implements basic functionality for CAMs.
    * `get_cams_generator()`
    * `get_cams_on_batch()`
        - Parameters: A list of image paths, or a batch of images with optional shapes list.
        - Returns: List of generated CAMs
    * `get_featuremap()` method in `CAM`.
        - Generates the featuremaps and softmax outputs.
        - Parameters: `model`, `model_layer`, and `batch_of_imgs`.
        - Returns: `featuremap`
    * `get_preds()` method in `CAM`:
        - Returns the softmax_prob and class of the predicted class.

    * `get_cams()` method in `CAM` to generate CAM.
        - Generates CAM for a batch of images.
        - Parameters: `featuremap`, `softmax_pred`, and `image_shapes`, 
        - Parameters: `multiprocessing` for multi-processing resize.
        - Returns: `cams` a list of cams generated.

## Benchmark time on CIFAR-10

## Create example notebooks


## Scale the library for Video - PHASE 2



