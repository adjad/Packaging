# First Neural Net
# Train, evaluate, and predict with the model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas
data = pandas.read_csv("Packaging Data - Training Data.csv")
labels=data.pop('Package Answer')
data.pop('Material type')
data.pop('Returns')
data.pop('Item #')
data.pop('No.')
tf.convert_to_tensor(data)
# mnist = keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)

# # normalize: 0,255 -> 0,1
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model
datatest = pandas.read_csv("Packaging Data - Test Data.csv")
labelstest=datatest.pop('Package Answer')
print(labelstest.dtype)
datatest.pop('Material type')
datatest.pop('Returns')
datatest.pop('Item #')
datatest.pop('No.')
tf.convert_to_tensor(datatest)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(109,)),
    keras.layers.Dense(600, activation='relu'),
    keras.layers.Dense(600, activation='relu'),
    keras.layers.Dense(16),
])

print(model.summary())

# another way to build the Sequential model:
#model = keras.models.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28,28))
#model.add(keras.layers.Dense(128, activation='relu'))
#model.add(keras.layers.Dense(10))

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 500

model.fit(data, labels, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
model.evaluate(datatest,labelstest, verbose=2)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(datatest)
print(np.argmax(predictions[0]))
print(labelstest[0])
model.save('the_model')
