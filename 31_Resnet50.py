# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:24:02 2019

@author: parveen
"""

from os.path import join
from keras.models import load_model
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Paths
data_path = 'T:/Projects/AI'
model = load_model(join(data_path, '_Models/model_image_classifier_resnet50.hdf5'))


img_path = join(data_path, 'elephant.jpg')
img = image.load_img(img_path, target_size=(224, 224))
image = Image.open(img)
img.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds) )
