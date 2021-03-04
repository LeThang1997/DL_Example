# Created by: Le Manh Thang
# Date: 04/03/2021

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# load data 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
val_images, val_labels = train_images[50000:60000,:], train_labels[50000:60000]
train_images, train_labels = train_images[:50000,:], train_labels[:50000]
#convert to float32
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0
# reshape to 28x28
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# one-hot encoding label
train_labels = np_utils.to_categorical(train_labels, 10)
val_labels = np_utils.to_categorical(val_labels, 10)
test_labels = np_utils.to_categorical(val_labels,10)

# create model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=1, batch_size=128)

score = model.evaluate(test_images, test_labels, verbose=0)
print(score)

model.save("mnist.h5")

print("Built model MNIST successfully\n")