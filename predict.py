import smash

import os

import numpy as np
import pandas as pd

from tools import model_to_df, feature_engineering, df_to_network_in, lstm_net


# = DEFINE CONSTANTS ==
# =====================

PATH_NET = "net"
PATH_FILEMODEL = "models/ardeche-test.hdf5"
PATH_FILEOUT = "res/predict-test.csv"

# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(PATH_FILEMODEL)
df = model_to_df(model, target_mode=False)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["timestep"].unique()
pred_set = df[~df.timestep.isin(missing)]

print(pred_set)

# % Normalize and prepare inputs for the network
pred, _ = df_to_network_in(pred_set, target_mode=False)

# = WRITE CORRECTED FILES ==
# ==========================

os.makedirs(os.path.dirname(PATH_FILEOUT), exist_ok=True)

k_fold = len([f for f in os.listdir(PATH_NET) if f.endswith(".index")])

nets = []

for fold in range(k_fold):
    net = lstm_net(pred.shape[-2:])
    net.load_weights(os.path.join(PATH_NET, f"fold_{fold + 1}"))
    nets.append(net)

y_pred = np.mean([net.predict(pred) for net in nets], axis=0)
df_pred = pd.DataFrame(
    {
        "code": pred_set["code"],
        "timestep": pred_set["timestep"],
        "bias": y_pred.flatten(),
    }
)
df_pred.to_csv(PATH_FILEOUT, index=False)
