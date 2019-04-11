# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:24:02 2019

@author: parveen
"""
from keras import backend as K
from os.path import join
from os.path import dirname
from keras.models import load_model
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Meta information
suffix = '_resnet50'
Service = 'ServiceImage'
ProjHome = 'T:/AI-Product/_Service'
ImagePath = join(dirname(ProjHome), 'zz_sampledata')

K.set_learning_phase(0)
model = load_model(join(ProjHome, Service + suffix + '.hdf5'))
# load and prepare image
img = image.load_img(join(ImagePath, 'african.jpg'), target_size=(224, 224))

img.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)
print(decode_predictions(preds))

###############################################################################

#Grad-CAM process
african_elephant_output = model.output[:, np.argmax(preds[0])]
last_conv_layer = model.get_layer('activation_49')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0,1,2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512) :
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

img = cv2.imread(join(ImagePath, 'african.jpg'))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('D:/Projects/Africanelephant_ResNet_cam.jpg', superimposed_img)
