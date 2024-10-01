import os 
import tensorflow as tf
import numpy as np
import imageio
from typing import List
from utils import mappable_function, num_to_char, char_to_num,physical_devices

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500,reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)
#Added for split
train= data.take(450)
test= data.skip(450)

frames,aligns = data.as_numpy_iterator().next()

#testing the data pipeline and preprocessning
test1= data.as_numpy_iterator()
val= test1.next()
fv = (val[0][1].astype(np.uint8) * 255).squeeze()
imageio.mimsave('./animation.gif', fv, duration=100)

