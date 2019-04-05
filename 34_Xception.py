# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:26:08 2019

@author: parveen
"""

from os.path import join
from os.path import dirname
from keras.models import load_model
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

# Meta information
suffix = '_Xception'
Service = 'ServiceImage'
ProjHome = 'T:/AI-Product/_Service'
ImagePath = join(dirname(ProjHome), 'zz_sampledata')

model = load_model(join(ProjHome, Service + suffix + '.hdf5'))

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
