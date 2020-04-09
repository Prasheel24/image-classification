from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("!------------------------------------------------------------------!")
print(x_train.shape, y_train.shape)
print("!------------------------------------------------------------------!")
print(x_test.shape, y_test.shape)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='rmsprop', #adam
              loss=loss_fn,
              metrics=['accuracy'])

hist = model.fit(x=x_train,y=y_train, epochs=20, validation_data=(x_test, y_test)) 


train_score = model.evaluate(x_train,  y_train, verbose=2)
test_score = model.evaluate(x_test,  y_test, verbose=2)

print("Train loss {:.4f}, accuracy {:.2f}%".format(train_score[0], train_score[1] * 100)) 
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100)) 


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

model.summary()