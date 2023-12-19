import smash

import os
import argparse

from tools import (
    model_to_df,
    feature_engineering,
    df_to_network_in,
    build_lstm,
    log_lkh,
)

import tensorflow as tf
from sklearn.model_selection import KFold

# % Check version and GPU
print("Tensorflow version: ", tf.__version__)

print(
    "GPU available: ",
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None),
)


# = ARGUMENT PARSER ==
# ====================

parser = argparse.ArgumentParser()

parser.add_argument(
    "-pm",
    "--path_filemodel",
    type=str,
    help="Select the smash Model object",
)

parser.add_argument(
    "-pn",
    "--path_netout",
    type=str,
    default=f"{os.getcwd()}/net",
    help="[optional] Select the output directory for the trained neural network",
)

parser.add_argument(
    "-gp",
    "--gaussian_parameters",
    type=str,
    default="std",
    choices=["mean", "std", "both"],
    help="[optional] Select the Gaussian parameters to be optimized",
)

parser.add_argument(
    "-ss",
    "--sequence_size",
    type=int,
    default=10,
    help="[optional] Select the squence size of inputs in LSTM network",
)

parser.add_argument(
    "-e",
    "--epoch",
    type=int,
    default=200,
    help="[optional] Select the number of epochs for training",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=512,
    help="[optional] Select the batch size for training",
)

parser.add_argument(
    "-k",
    "--kfold",
    type=int,
    default=4,
    help="[optional] Select the number of folds for cross-validation",
)

parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="adam",
    help="[optional] Select the optimization algorithm",
)

parser.add_argument(
    "-ilr",
    "--init_lr",
    type=float,
    default=1e-3,
    help="[optional] Select the value of initial learning rate",
)

parser.add_argument(
    "-l",
    "--loss",
    type=str,
    default="log_lkh",
    choices=["log_lkh", "mse", "mae"],
    help="[optional] Select the loss function to train the neural network",
)

args = parser.parse_args()
if args.loss == "log_lkh":
    if args.gaussian_parameters == "mean":
        raise ValueError(
            "Cannot maximize the log likelihood function when std=0 (or gaussian_parameters=='mean')"
        )
else:
    if args.gaussian_parameters == "std":
        raise ValueError(
            f"zero mean (or gaussian_parameters=='std') is not relevant for minimizing the {args.loss} loss function"
        )

# gauge_train = ["K2064010", "V5015210", "V5004030"]


# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(args.path_filemodel)
# df = model_to_df(model, args.sequence_size, target_mode=True, gauge=gauge_train)
df = model_to_df(model, args.sequence_size, target_mode=True)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["id"].unique()
train_set = df[~df.id.isin(missing)]

# % Normalize and prepare inputs for the network
output_size = 2 if args.gaussian_parameters == "both" else 1
train, target = df_to_network_in(
    train_set, args.sequence_size, output_size, target_mode=True
)

# = TRAINING ==
# =============

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    kf = KFold(n_splits=args.kfold, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(train, target)
    ):  # Cross Validation Training
        print(f"</> Training Fold {fold + 1}...")

        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]

        net = build_lstm(train.shape[-2:], output_size)
        net.compile(
            optimizer=args.optimizer,
            loss=log_lkh if args.loss == "log_lkh" else args.loss,
        )

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            args.init_lr, 100 * ((train.shape[0] * 0.8) / args.batch_size), 1e-5
        )
        lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        cp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.path_netout, f"fold_{fold + 1}"),
            save_weights_only=True,
            mode="min",
            save_best_only=True,
        )

        net.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=args.epoch,
            batch_size=args.batch_size,
            callbacks=[lr, cp],
        )
