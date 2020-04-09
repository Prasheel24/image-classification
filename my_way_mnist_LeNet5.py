import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import utils
import sys, os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# set numeric type to float32 from unit8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train, x_test = x_train / 255.0, x_test / 255.0

# Transform lables to one-hot encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1) 

#Instantiate an empty model
model = tf.keras.models.Sequential()

# C1 Convolutional Layer
model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding="same"))

# S2 Pooling Layer
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding=’valid’))

# C3 Convolutional Layer
model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding=’valid’))

# C5 Fully Connected Convolutional Layer
model.add(tf.keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(tf.keras.layers.Flatten())

# FC6 Fully Connected Layer
model.add(tf.keras.layers.Dense(84, activation='tanh'))

#Output Layer with softmax activation
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"]) 
# model.summary()

hist = model.fit(x=x_train,y=y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=1) 

train_score = model.evaluate(x_train, y_train, verbose=1)
test_score = model.evaluate(x_test, y_test, verbose=1)

print("Train loss {:.4f}, accuracy {:.2f}%".format(train_score[0], train_score[1] * 100)) 
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100)) 

model.summary()

f, ax = plt.subplots()
ax.plot([None] + hist.history['loss'], 'o-')
ax.plot([None] + hist.history['val_loss'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train Loss', 'Test Loss'], loc = 0)
ax.set_title('Training/Test Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss') 
# plt.plot()
plt.show()