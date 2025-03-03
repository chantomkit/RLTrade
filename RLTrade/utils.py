import numpy as np
import pandas as pd

class FeatureEngineering:
    def __init__(self, df, main_feature_col="close"):
        self.df = df.copy()
        self.main_feature_col = main_feature_col

    def build_features(self, additonal_features=[]):
        feature_cols = [self.main_feature_col]
        if additonal_features:
            feature_cols += additonal_features
        feat = self.df[feature_cols].add_prefix("feature_")
        self.df = pd.concat([self.df, feat], axis=1)

    def rolling_feature(self, window=24, mode="default", subtract_mean=False):
        df_copy = self.df.copy()
        if mode == "default":
            target = df_copy[self.main_feature_col]
        elif mode == "log":
            target = np.log(df_copy[self.main_feature_col])
        elif mode == "diff":
            target = df_copy[self.main_feature_col].diff().fillna(0)
        elif mode == "logdiff":
            target = np.log(df_copy[self.main_feature_col]).diff().fillna(0)
        elif mode == "pct_change":
            target = df_copy[self.main_feature_col].pct_change().fillna(0)
        elif mode == "logpct_change":
            target = np.log(df_copy[self.main_feature_col].pct_change().fillna(1))

        rolling_features = pd.DataFrame(index=self.df.index)
        for i in range(window):
            rolling_features[f"feature_rolling_{i}"] = target.shift(i)
            
        if subtract_mean:
            rolling_features = rolling_features - rolling_features.mean(axis=1).values.reshape(-1, 1)
        rolling_features = rolling_features.ffill(axis=1)

        df_copy = pd.concat([df_copy, rolling_features], axis=1)
        self.df = df_copy
    
    def rolling_mean_corrected_feature(self, window=24, mode="default"):
        df_copy = self.df.copy()
        if mode == "default":
            target = df_copy[self.main_feature_col]
        elif mode == "log":
            target = np.log(df_copy[self.main_feature_col])
        elif mode == "diff":
            target = df_copy[self.main_feature_col].diff().fillna(0)
        elif mode == "logdiff":
            target = np.log(df_copy[self.main_feature_col]).diff().fillna(0)

        rolling_means = target.rolling(window=window, min_periods=1).mean()
        target = target - rolling_means
        df_copy[f"feature_rolling_mean_corrected"] = target
        self.df = df_copy
    
# def build_features(df, features_col="close"):
#     feat = df[[features_col]].add_prefix("feature_")
#     return pd.concat([df, feat], axis=1)

# def rolling_feature(df, feature_col="close", window=24, mode="default", subtract_mean=False):
#     df_copy = df.copy()
#     if mode == "default":
#         target = df_copy[feature_col]
#     elif mode == "log":
#         target = np.log(df_copy[feature_col])
#     elif mode == "diff":
#         target = df_copy[feature_col].diff().fillna(0)
#     elif mode == "logdiff":
#         target = np.log(df_copy[feature_col]).diff().fillna(0)

#     rolling_features = pd.DataFrame(index=df.index)
#     for i in range(window):
#         rolling_features[f"feature_rolling_{i}"] = target.shift(i)
#     rolling_features = rolling_features.bfill(axis=0)
#     if subtract_mean:
#         rolling_features = rolling_features - rolling_features.mean(axis=1).values.reshape(-1, 1)

#     df_copy = pd.concat([df_copy, rolling_features], axis=1)
#     return df_copy

# def rolling_mean_corrected_feature(df, feature_col="close", window=24, mode="default"):

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