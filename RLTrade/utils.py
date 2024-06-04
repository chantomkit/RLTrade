import numpy as np
import pandas as pd

def build_features(df, features_col="close"):
    feat = df[[features_col]].add_prefix("feature_")
    return pd.concat([df, feat], axis=1)

def build_rolling_feature(df, feature_col="close", window=24, mode="default", mean_correction=False):
    df_copy = df.copy()
    if mode == "default":
        target = df_copy[feature_col]
    elif mode == "log":
        target = np.log(df_copy[feature_col])
    elif mode == "diff":
        target = df_copy[feature_col].diff().fillna(0)
    elif mode == "logdiff":
        target = np.log(df_copy[feature_col]).diff().fillna(0)

    rolling_features = pd.DataFrame(index=df.index)
    for i in range(window):
        rolling_features[f"feature_rolling_{i}"] = target.shift(i)
    rolling_features = rolling_features.bfill(axis=0)
    if mean_correction:
        rolling_features = rolling_features - rolling_features.mean(axis=1).values.reshape(-1, 1)

    df_copy = pd.concat([df_copy, rolling_features], axis=1)
    return df_copy

def stationaryDGP(
        N=10000,
        sigmaX = 0.05,
        sigmaEta = 0.1,
        theta = 0.1,
        mu = 100.,
    ):
    # We have 2 cointegrated time series X and Y, related by some constant relationship, whose
    # successful estimation is necessary for optimal trading of assets X and Y.
    X = []
    Y = []
    epsilon = [mu]
    for t in range(N):
        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(epsilon[-1] + theta * (mu - epsilon[-1]) + np.random.normal(0., sigmaEta))

        Y.append(X[-1] + epsilon[-1])
    return np.array(X), np.array(Y)

def nonstationaryDGP(
        N=10000,
        sigmaX = 0.05,
        sigmaEta = 0.1,
        theta = 0.05,
        mu = 100.,
    ):

    X = []
    Y = []
    epsilon = [mu]
    for t in range(N):
        if t % 200 == 0:
            mu += np.random.standard_cauchy() * 0.01

        if np.random.uniform(0, 1) >= 0.995:
            mu += np.random.uniform(-3, 3)

        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(epsilon[-1] + theta * (mu - epsilon[-1]) + np.random.normal(0., sigmaEta))

        Y.append(X[-1] + epsilon[-1])
    return np.array(X), np.array(Y)