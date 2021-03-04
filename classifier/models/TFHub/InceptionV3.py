# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: train.py
# Author: ThangLM
# Created: 28/12/2020
# Description:

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import tensorflow as tf

def inceptionv3(classes):
    modelInception = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001], trainable=True)
    modelInception.build([None, 299, 299, 3])  # Batch input shape.
    x = tf.keras.layers.Flatten()(modelInception.output)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(modelInception.input, x)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
    return model