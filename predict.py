import smash

import os

import numpy as np
import pandas as pd

from tools import model_to_df, feature_engineering, df_to_network_in, lstm_net


# = PATH FILES ==
# ===============

PATH_MODEL = "models/ardeche-test.hdf5"
PATH_NET = "net"
PATH_PREDICT = "res-predict"

# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(PATH_MODEL)
df = model_to_df(model, target_present=False)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["timestep"].unique()
pred_set = df[~df.timestep.isin(missing)]

print(pred_set)

# % Normalize and prepare inputs for the network
pred, _ = df_to_network_in(pred_set, target_present=False)

# = WRITE CORRECTED FILES ==
# ==========================

if not os.path.exists(PATH_PREDICT):
    os.makedirs(PATH_PREDICT)

k_fold = len([f for f in os.listdir(PATH_NET) if f.endswith(".index")])

nets = []

for fold in range(k_fold):
    net = lstm_net(pred.shape[-2:])
    net.load_weights(f"{PATH_NET}/fold_{fold + 1}")
    nets.append(net)

y_pred = np.mean([net.predict(pred) for net in nets], axis=0)
df_pred = pd.DataFrame(
    {
        "code": pred_set["code"],
        "timestep": pred_set["timestep"],
        "bias": y_pred.flatten(),
    }
)
df_pred.to_csv(f"{PATH_PREDICT}/pred.csv", index=False)
