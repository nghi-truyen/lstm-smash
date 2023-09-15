import smash

import os

import numpy as np
import pandas as pd

from tools import model_to_df, feature_engineering, df_to_network_in, lstm_net

import tensorflow as tf
from sklearn.model_selection import KFold


# % Check version and GPU
print("Tensorflow version: ", tf.__version__)

print(
    "GPU avialable: ",
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None),
)

# = DEFINE CONSTANTS ==
# =====================

# % Paths
PATH_FILEMODEL = "models/ardeche-train.hdf5"
PATH_NETOUT = "net"

# % Training parameters
EPOCH = 200
BATCH_SIZE = 512
K_FOLD = 5

OPTIMIZER = "adam"
LOSS = "mse"

# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(PATH_FILEMODEL)
df = model_to_df(model)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["timestep"].unique()
train_set = df[~df.timestep.isin(missing)]

# % Normalize and prepare inputs for the network
train, target = df_to_network_in(train_set)

# = TRAINING ==
# =============

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    kf = KFold(n_splits=K_FOLD, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(train, target)
    ):  # Cross Validation Training
        print(f"</> Training Fold {fold + 1}...")

        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]

        net = lstm_net(train.shape[-2:])
        net.compile(optimizer=OPTIMIZER, loss=LOSS)

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-3, 100 * ((train.shape[0] * 0.8) / BATCH_SIZE), 1e-5
        )
        lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        cp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PATH_NETOUT, f"fold_{fold + 1}"),
            save_weights_only=True,
            mode="min",
            save_best_only=True,
        )

        net.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            callbacks=[lr, cp],
        )
