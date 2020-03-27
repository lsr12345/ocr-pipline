
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import os
import json

from tools.network import East, CRNN, predict_det, predict_rec

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[ ]:


with open('./tools/id2char.json', 'r') as f:
    id2char = json.load(f)


# In[ ]:


ade_model_path = './models/east_model_weights_3T512.h5'
crnn_model_path = './models/crnn_efficientnetb7_val_loss.0.95.h5'

east = East()
east_model = east.east_network()
print('Construct east_model Done!')
east_model.load_weights(ade_model_path)
print('Load weights Done!')

crnn = CRNN()
crnn_model = crnn.crnn_network()
print('Construct crnnt_model Done!')
crnn_model.load_weights(crnn_model_path)
print('Load weights Done!')


# In[ ]:


img_path = './demo/2.jpg'
quad_scores, quad_after_nms, d_wight, d_height = predict_det(east_model, img_path)
results, coordinate = predict_rec(crnn_model, img_path, quad_scores, quad_after_nms, d_wight, d_height, id2char)

