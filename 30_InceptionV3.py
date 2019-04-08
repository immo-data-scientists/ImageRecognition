# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:22:02 2019

@author: parveen
"""

from os.path import join
from os.path import dirname
from keras.models import load_model
from keras.preprocessing import image
from imagenet_utils import decode_predictions
import numpy as np

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# Meta information
suffix = '_InceptionV3'
Service = 'ServiceImage'
ProjHome = 'T:/AI-Product/_Service'
ImagePath = join(dirname(ProjHome), 'zz_sampledata')

model = load_model(join(ProjHome, Service + suffix + '.hdf5'))

#model = InceptionV3(weights='imagenet')


# load and prepare image
img = image.load_img(join(ImagePath, 'elephant.jpg'), target_size=(299, 299))
#image = Image.open(img)

img.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)
print('Predicted:', decode_predictions(preds) )