import smash
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from datetime import timedelta


def model_to_df(
    model: smash.Model,
    sequence_size: int,
    target_mode: bool = True,
    precip: bool = True,
    pot_evapot: bool = True,
    precip_ind: bool = True,
    gauge: list | None = None,
):
    """
    Read Model object in smash and extract into a raw DataFrame.
    """
    dict_df = {}

    # % ID
    n_sequence = model.setup.ntime_step // sequence_size

    ntime_step = model.setup.ntime_step - (model.setup.ntime_step % sequence_size)

    id_single_gauge = np.repeat(np.arange(n_sequence), sequence_size)
    dict_df["id"] = np.concatenate(
        [id_single_gauge + i * n_sequence for i in range(model.mesh.ng)]
    )

    # % Catchment code
    dict_df["code"] = np.repeat(model.mesh.code, ntime_step)

    # % Timestep
    dict_df["timestep"] = np.tile(np.arange(ntime_step), model.mesh.ng)

    # % Meaningful timestep in year for learning
    tsy = _timestep_convert(model.setup.start_time, model.setup.dt, ntime_step)
    dict_df["timestep_in_year"] = np.tile(tsy, model.mesh.ng)

    # % Simumated discharges
    qs = model.response.q[..., :ntime_step]
    qs[qs < 0] = 0
    dict_df["discharge_sim"] = qs.flatten(order="C")

    # % Bias
    if target_mode:
        qo = model.response_data.q[..., :ntime_step]
        qo[qo < 0] = np.nan
        bias = qo - qs
        dict_df["bias"] = bias.flatten(order="C")
        dict_df["std_bias"] = np.zeros(bias.size)

    # % Mean precipitation
    if precip:
        prcp = model.atmos_data.mean_prcp[..., :ntime_step]
        prcp[prcp < 0] = np.nan
        dict_df["precipitation"] = prcp.flatten(order="C")

    # % PET
    if pot_evapot:
        pet = model.atmos_data.mean_pet[..., :ntime_step]
        pet[pet < 0] = np.nan
        dict_df["pet"] = pet.flatten(order="C")

    # % Precipitation indices
    if precip_ind:
        prcp_ind = smash.precipitation_indices(model)
        d1 = prcp_ind.d1[..., :ntime_step]
        d1[np.isnan(d1)] = -1
        d2 = prcp_ind.d2[..., :ntime_step]
        d2[np.isnan(d2)] = -1
        dict_df["d1"] = d1.flatten(order="C")
        dict_df["d2"] = d2.flatten(order="C")

    df = pd.DataFrame(dict_df)

    if not gauge is None:
        df = df[df["code"].isin(gauge)]

    return df


def _timestep_convert(st, dt, n_ts, by=None):
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


def df_to_network_in(df: pd.DataFrame, sequence_size: int, target_mode: bool = True):
    """
    Normalize data and prepare input for the neural network.
    """
    # % Drop info columns
    df = df.drop(["id", "code", "timestep"], axis=1)

    if target_mode:
        # check if 'bias' and 'std_bias' are already located in the last 2 columns
        if df.columns[-2] != "bias" or df.columns[-1] != "std_bias":
            columns = [col for col in df.columns if not "bias" in col]
            columns = np.append(columns, ("bias", "std_bias"))
            df = df[columns]

        # convert to numpy array
        data = df.to_numpy()[..., :-2]
        target = df.to_numpy()[..., -2:]
        target = target.reshape(-1, sequence_size, 2)

    else:
        data = df.to_numpy()
        target = None

    # % Normalize
    data = RobustScaler().fit_transform(data)
    data = data.reshape(-1, sequence_size, data.shape[-1])

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
                # recurrent_regularizer=tf.keras.regularizers.l2(6e-3),
                return_sequences=True,
            )
        )
    )
    # net.add(
    #     tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(
    #             64,
    #             activation="relu",
    #             recurrent_regularizer=tf.keras.regularizers.l2(6e-3),
    #             return_sequences=True,
    #         )
    #     )
    # )
    net.add(tf.keras.layers.Dense(32, activation="selu"))
    # net.add(tf.keras.layers.Dropout(0.1))
    net.add(tf.keras.layers.Dense(2))

    return net
