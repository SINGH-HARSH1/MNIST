import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
# import matplotlib.pyplot as plt


mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print(y_train_full.dtype)

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

model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(0.02), metrics=["accuracy"])

lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5)
checkpoint_cb = ModelCheckpoint("Mnist_model1.h5", save_best_only=True)
earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid),
                    callbacks=[lr_scheduler, checkpoint_cb, earlystopping_cb, tensorboard_cb])


model.evaluate(X_test, y_test)
