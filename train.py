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

# % Path file
PATH_MODEL_CAL = "models/ardeche-train.hdf5"
PATH_MODEL_VAL = "models/ardeche-test.hdf5"

PATH_NETOUT = "net"

PATH_CORRECTED_FILEOUT = "res"

# % Training parameters
EPOCH = 200
BATCH_SIZE = 512
K_FOLD = 5

OPTIMIZER = "adam"
LOSS = "mse"

# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model_train = smash.io.read_model(PATH_MODEL_CAL)
df_train = model_to_df(model_train)
df_train = feature_engineering(df_train)

model_test = smash.io.read_model(PATH_MODEL_VAL)
df_test = model_to_df(model_test)
df_test = feature_engineering(df_test)

# % Handle missing data
missing_train = df_train[df_train.isna().any(axis=1)]["timestep"].unique()
train_set = df_train[~df_train.timestep.isin(missing_train)]

missing_test = df_test[df_test.isna().any(axis=1)]["timestep"].unique()
test_set = df_test[~df_test.timestep.isin(missing_test)]

# % Normalize and prepare inputs for the network
train, target = df_to_network_in(train_set)
test, target_test = df_to_network_in(test_set)

# = TRAINING ==
# =============

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    kf = KFold(n_splits=K_FOLD, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(
        kf.split(train, target)
    ):  # Cross Validation Training
        print(f"</> Training Fold {fold + 1}...")

        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = target[train_idx], target[test_idx]

        net = lstm_net(train.shape[-2:])
        net.compile(optimizer=OPTIMIZER, loss=LOSS)

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-3, 100 * ((train.shape[0] * 0.8) / BATCH_SIZE), 1e-5
        )
        lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        cp = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{PATH_NETOUT}/fold_{fold + 1}",
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

# = WRITE CORRECTED FILES ==
# ==========================

if not os.path.exists(PATH_CORRECTED_FILEOUT):
    os.makedirs(PATH_CORRECTED_FILEOUT)

nets = []

for fold in range(K_FOLD):
    net = lstm_net(train.shape[-2:])
    net.load_weights(f"{PATH_NETOUT}/fold_{fold + 1}")
    nets.append(net)

ypred_train = np.mean([net.predict(train) for net in nets], axis=0)
df_train_correct = pd.DataFrame(
    {
        "code": train_set["code"],
        "timestep": train_set["timestep"],
        "bias": ypred_train.flatten(),
    }
)
df_train_correct.to_csv(f"{PATH_CORRECTED_FILEOUT}/corrected-train.csv", index=False)

ypred_test = np.mean([net.predict(test) for net in nets], axis=0)
df_test_correct = pd.DataFrame(
    {
        "code": test_set["code"],
        "timestep": test_set["timestep"],
        "bias": ypred_test.flatten(),
    }
)
df_test_correct.to_csv(f"{PATH_CORRECTED_FILEOUT}/corrected-test.csv", index=False)
