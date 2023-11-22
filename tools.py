import smash
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from datetime import timedelta


def model_to_df(
    model: smash.Model,
    target_mode: bool = True,
    precip: bool = True,
    pot_evapot: bool = True,
    precip_ind: bool = True,
    gauge: bool = None,
):
    """
    Read Model object in smash and extract into a raw DataFrame.
    """
    dict_df = {}

    # % Catchment code for info
    code = model.mesh.code
    code = np.tile(code, model.setup.ntime_step)
    dict_df["code"] = code

    # % Timestep for info
    timestep = np.repeat(np.arange(model.setup.ntime_step), model.mesh.ng)
    dict_df["timestep"] = timestep

    # % Meaningful timestep in year for learning
    tsy = _timestep_1year(
        model.setup.start_time, model.setup.dt, model.setup.ntime_step
    )
    tsy = np.repeat(tsy, model.mesh.ng)
    dict_df["timestep_in_year"] = tsy

    # % Simumated discharges
    qs = model.response.q
    dict_df["discharge_sim"] = qs.flatten(order="F")

    # % Bias
    if target_mode:
        qo = model.response_data.q.copy()
        qo[qo < 0] = np.nan
        bias = qo - qs
        dict_df["bias"] = bias.flatten(order="F")
        dict_df["std_bias"] = np.zeros(bias.size)

    # % Mean precipitation
    if precip:
        prcp = model.atmos_data.mean_prcp.copy()
        prcp[prcp < 0] = np.nan
        dict_df["precipitation"] = prcp.flatten(order="F")

    # % PET
    if pot_evapot:
        pet = model.atmos_data.mean_pet.copy()
        pet[pet < 0] = np.nan
        dict_df["pet"] = pet.flatten(order="F")

    # % Precipitation indices
    if precip_ind:
        prcp_ind = smash.precipitation_indices(model)
        d1 = prcp_ind.d1.copy()
        d1[np.isnan(d1)] = -1
        d2 = prcp_ind.d2.copy()
        d2[np.isnan(d2)] = -1
        dict_df["d1"] = d1.flatten(order="F")
        dict_df["d2"] = d2.flatten(order="F")

    df = pd.DataFrame(dict_df)

    if not gauge is None:
        df = df[df["code"].isin(gauge)]

    return df


def _timestep_1year(st, dt, n_ts, by=None):
    if by is None:
        by = 1
    elif isinstance(by, str):
        if by == "hour":
            by = int(60 * 60 / dt)
        elif by == "day":
            by = int(24 * 60 * 60 / dt)
        elif by == "month":
            by = int(365 / 12 * 24 * 60 * 60 / dt)

    timestep = np.arange(1, int(365 * 24 * 60 * 60 / dt) + 1)

    defst = f"{pd.to_datetime(st).year}-08-01 00:00:00"

    if pd.Timestamp(st) < pd.Timestamp(defst):
        timestamps = pd.date_range(start=st, end=defst, freq=timedelta(seconds=dt))
        s_ind = timestep.size - (len(timestamps) - 1)
    else:
        timestamps = pd.date_range(start=defst, end=st, freq=timedelta(seconds=dt))
        s_ind = len(timestamps) - 1

    return np.array([timestep[(s_ind + i) % len(timestep)] // by for i in range(n_ts)])


def feature_engineering(df: pd.DataFrame):
    """
    Perform feature engineering from the raw DataFrame.
    """
    df["year"] = df["timestep"] // np.max(df["timestep_in_year"])
    drop_cols = ["year"]

    df["discharge_sim_cumsum"] = df.groupby(["code", "year"])["discharge_sim"].cumsum()

    try:
        df["precipitation_cumsum"] = df.groupby(["code", "year"])[
            "precipitation"
        ].cumsum()
    except:
        pass

    try:
        df["pet_cumsum"] = df.groupby(["code", "year"])["pet"].cumsum()
        df["sqrt_pet"] = np.sqrt(df["pet"])
        drop_cols.append("pet")
    except:
        pass

    df = df.drop(drop_cols, axis=1)

    return df


def df_to_network_in(df: pd.DataFrame, target_mode: bool = True):
    """
    Normalize data and prepare input for the neural network.
    """
    n_catch = df["code"].unique().size

    # % Drop info columns
    df = df.drop(["code", "timestep"], axis=1)

    if target_mode:
        # check if 'bias' and 'std_bias' are already located in the last 2 columns
        if df.columns[-2] != "bias" or df.columns[-1] != "std_bias":
            columns = [col for col in df.columns if not "bias" in col]
            columns = np.append(columns, ("bias", "std_bias"))
            df = df[columns]

        # convert to numpy array
        data = df.to_numpy()[..., :-2]
        target = df.to_numpy()[..., -2:]
        target = target.reshape(-1, n_catch, 2)

    else:
        data = df.to_numpy()
        target = None

    # % Normalize
    data = RobustScaler().fit_transform(data)
    data = data.reshape(-1, n_catch, data.shape[-1])

    return data, target


def log_lkh(y_true, y_pred):
    return -tf.reduce_mean(
        tf.math.log(1 / (tf.abs(y_pred[..., 1]) * tf.sqrt(2 * np.pi)))
        - 0.5 * tf.square((y_true[..., 0] - y_pred[..., 0]) / y_pred[..., 1])
    )


def build_lstm(input_shape):
    """
    The LSTM neural network for learning streamflow prediction error.
    """
    net = tf.keras.Sequential()

    net.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                input_shape=input_shape,
                activation="relu",
                recurrent_regularizer=tf.keras.regularizers.l2(6e-3),
                return_sequences=True,
            )
        )
    )
    net.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                64,
                activation="relu",
                recurrent_regularizer=tf.keras.regularizers.l2(6e-3),
                return_sequences=True,
            )
        )
    )
    net.add(tf.keras.layers.Dense(32, activation="selu"))
    net.add(tf.keras.layers.Dropout(0.1))
    net.add(tf.keras.layers.Dense(2))

    return net
