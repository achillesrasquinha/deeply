# import numpy as np
# import imgaug.augmenters as iaa
# import cv2

# def func_images(images, *args, **kwargs):
#     results = []
#     for image in images:
#         dilated = cv2.dilate(image)
#         results.append(image)
#     return results

# def func_heatmaps(heatmaps, *args, **kwargs):
#     return heatmaps
    
# def func_keypoints(keypoints_on_images, *args, **kwargs):
#     return keypoints_on_images

# def Dilate(kernel = np.ones((1, 1)), iterations = 1):
#     return iaa.Lambda(
#         func_images    = func_images,
#         func_heatmaps  = func_heatmaps,
#         func_keypoints = func_keypoints
#     )

import numpy as np
import cv2

from imgaug.augmenters import Augmenter
# from imgaug.parameters import handle_probability_param
import imgaug.parameters as iap
import imgaug as ia

# https://github.com/aleju/imgaug/issues/128
class Dilate(Augmenter):
    def __init__(self, kernel = np.ones((1, 1)), iterations = 1, *args, **kwargs):
        self._super = super(Dilate, self)
        self._super.__init__(*args, **kwargs)
        
        self.kernel     = kernel
        self.iterations = iterations
        
    def _augment_batch_(self, batch, *args, **kwargs):
        if batch.images is None:
            return batch

        images = batch.images

        for i, image in enumerate(images):
            batch.images[i] = cv2.dilate(image, self.kernel, iterations = self.iterations)

        return batch

    def get_parameters(self):
        return [self.kernel, self.iterations]