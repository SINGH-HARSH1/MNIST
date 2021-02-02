"""
This Selu activation Function with normal distribution and comparing it with the  earlier default Relu model.
This activation layer self normalizes the data for wide and deep networks.
"""
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, AlphaDropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping


mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()


# Splitting the Train Set into Train And Validation Sets
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Normalizing the Datasets.
X_train = X_train/255.0
X_valid = X_valid/255.0
X_test = X_test/255.0


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Standardising the dataset for the selu activation Function
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)

X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

# model = Sequential()
# model.add(Flatten(input_shape=[28, 28]))
# model.add(AlphaDropout(rate=0.3))
# model.add(Dense(300, activation="selu",
#                 kernel_initializer="lecun_normal"))
# for layer in range(99):
#     model.add(AlphaDropout(rate=0.3))
#     model.add(Dense(300, activation="selu", kernel_initializer="lecun_normal"))  # Just trying a very deep network
#
# model.add(AlphaDropout(rate=0.3))
# model.add(Dense(10, activation="softmax"))


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 25
print(model.summary())
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5)
earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
n_epochs = 25
tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler, earlystopping_cb, tensorboard_cb])


model.evaluate(X_test_scaled, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
