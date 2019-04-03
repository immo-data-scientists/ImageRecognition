# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:25:05 2019

@author: parveen
"""

from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

img_path = 'ball.jfif'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))