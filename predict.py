import smash

import os
import argparse

import numpy as np
import pandas as pd

from tools import model_to_df, feature_engineering, df_to_network_in, lstm_net

import tensorflow as tf

# % Check version and GPU
print("Tensorflow version: ", tf.__version__)

print(
    "GPU avialable: ",
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None),
)


# = ARGUMENT PARSER ==
# ====================

parser = argparse.ArgumentParser()

parser.add_argument(
    "-pm",
    "--path_filemodel",
    type=str,
    help="Select the smash Model object to correct",
)

parser.add_argument(
    "-pn",
    "--path_net",
    type=str,
    help="Select the trained neural network to correct the Model object",
)

parser.add_argument(
    "-po",
    "--path_fileout",
    type=str,
    default=f"{os.getcwd()}/bias-pred.csv",
    help="[optional] Select path for the output file",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=512,
    help="[optional] Select the batch size for predicting",
)

args = parser.parse_args()

# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(args.path_filemodel)
df = model_to_df(model, target_mode=False)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["timestep"].unique()
pred_set = df[~df.timestep.isin(missing)]

# % Normalize and prepare inputs for the network
pred, _ = df_to_network_in(pred_set, target_mode=False)

# = PREDICT ==
# ============

try:
    os.makedirs(os.path.dirname(args.path_fileout), exist_ok=True)
except:
    pass

k_fold = len([f for f in os.listdir(args.path_net) if f.endswith(".index")])

nets = []

# % Predict with tf session
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    for fold in range(k_fold):
        net = lstm_net(pred.shape[-2:])
        net.load_weights(os.path.join(args.path_net, f"fold_{fold + 1}"))
        nets.append(net)

    y_pred = np.mean(
        [net.predict(pred, batch_size=args.batch_size) for net in nets], axis=0
    )

# % Write results to csv file
df_pred = pd.DataFrame(
    {
        "code": pred_set["code"],
        "timestep": pred_set["timestep"],
        "bias": y_pred.flatten(),
    }
)
df_pred = df_pred.pivot(index="timestep", columns="code", values="bias").reset_index()
df_pred.to_csv(args.path_fileout, index=False)
