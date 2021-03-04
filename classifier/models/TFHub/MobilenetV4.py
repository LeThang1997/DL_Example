

# Copyright (C) 2020, LE MANH THANG. All rights reserved
# Module: train.py
# Author: ThangLMb
# Created: 28/12/2020
# Description:

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import tensorflow as tf

def mobilenetV4(classes):
    mobilenetv4 = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/4",
                trainable=True, arguments=dict(batch_norm_momentum=0.997))
    ])
    # x = tf.keras.layers.Dropout(0.5)(mobilenetv4.output)
    # x = tf.keras.layers.Dense(classes)(x)
    # x = tf.keras.layers.Softmax()(x)
    # model = tf.keras.models.Model(mobilenetv4.input, x)
    mobilenetv4.build([None, 128, 128, 3])  # Batch input shape.
    mobilenetv4.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
    return mobilenetv4