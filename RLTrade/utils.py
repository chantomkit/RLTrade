import numpy as np
import pandas as pd


class FeatureEngineering:
    """
    A flexible feature-engineering pipeline for time series DataFrame.
    Build a sequence of feature transformations via a configuration list.
    """

    def __init__(self, df, feature_prefix="feature_"):
        self.df = df.copy()
        self.prefix = feature_prefix
        self._builders = {
            "select": self._select_features,
            "rolling": self._rolling_window,
            "rolling_mean_corrected": self._rolling_mean_corrected,
        }

    @staticmethod
    def _transform(series, mode):
        """
        Apply a transformation to a pandas Series.
        Supported modes: 'default', 'log', 'diff', 'logdiff', 'pct_change', 'logpct_change'
        """
        if mode == "default":
            return series
        if mode == "log":
            return np.log(series)
        if mode == "diff":
            return series.diff().fillna(0)
        if mode == "logdiff":
            return np.log(series).diff().fillna(0)
        if mode == "pct_change":
            return series.pct_change().fillna(0)
        if mode == "logpct_change":
            return np.log(series).pct_change().fillna(1)
        raise ValueError(f"Unsupported mode: {mode}")

    def build_features(self, steps):
        """
        Apply a sequence of feature-building steps.

        Parameters:
        - steps: list of dicts, each with:
            - 'type': one of 'select', 'rolling', 'rolling_mean_corrected'
            - 'params': dict of parameters for that builder

        Returns:
        - DataFrame with new features added
        """
        df = self.df
        for step in steps:
            ftype = step["type"]
            params = step.get("params", {})
            builder = self._builders.get(ftype)
            if not builder:
                raise ValueError(f"Unknown feature type: {ftype}")
            df = builder(df, **params)
        self.df = df
        return df

    def _select_features(self, df, cols, mode="default"):
        """Select original columns and add them with feature prefix."""
        selected = df[cols].add_prefix(self.prefix)
        return pd.concat([df, selected], axis=1)

    def _rolling_window(self, df, cols, window=24, mode="default", subtract_mean=False):
        """Compute rolling-lag features for given columns."""
        result = df.copy()
        for col in cols:
            series = self._transform(df[col], mode)
            # build lagged columns
            lags = [series.shift(i) for i in range(window)]
            rolling = pd.concat(lags, axis=1)
            rolling.columns = [f"{self.prefix}{col}_lag_{i}" for i in range(window)]
            if subtract_mean:
                rolling = rolling.sub(rolling.mean(axis=1), axis=0)
            rolling = rolling.ffill().fillna(0)
            result = pd.concat([result, rolling], axis=1)
        return result

    def _rolling_mean_corrected(self, df, cols, window=24, mode="default"):
        """Subtract rolling mean from transformed series for given columns."""
        result = df.copy()
        for col in cols:
            series = self._transform(df[col], mode)
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            corrected = series - roll_mean
            result[f"{self.prefix}{col}_rolling_mean_corrected"] = corrected
        return result


def stationary_prob_model(
    N=10000,
    sigma_x=0.05,
    sigma_eta=0.1,
    theta=0.1,
    mu=100.0,
    x0=10.0,
    random_state=None,
):
    """
    Generates two cointegrated time series X and Y using a stationary mean reverting error by OU process.

    Args:
        N (int): Number of time steps.
        sigma_x (float): Std deviation of X's random walk noise.
        sigma_eta (float): Noise amplitude of epsilon.
        theta (float): Friction parameter, speed of mean reversion of epsilon.
        mu (float): Long-run mean of the error process.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple: Arrays (X, Y) of shape (N,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X, epsilon = np.zeros(N), np.zeros(N)

    # Initial conditions
    X[0] = np.random.normal(x0, sigma_x)
    epsilon[0] = mu + np.random.normal(
        0.0, sigma_eta
    )  # Random initial value for epsilon

    for t in range(1, N):
        X[t] = X[t - 1] + np.random.normal(0.0, sigma_x)  # Random walk

        # Euler–Maruyama step for epsilon: ε_t = ε_{t-1} + θ(μ - ε_{t-1}) + η_t,   η_t∼𝒩(0,σ_{η}^2)
        # Assume Δt = 1 for simplicity
        epsilon[t] = (
            epsilon[t - 1]
            + theta * (mu - epsilon[t - 1])
            + np.random.normal(0.0, sigma_eta)
        )

    return X, X + epsilon  # Y = X + ε


def nonstationary_prob_model(
    N=10000,
    sigma_x=0.05,
    sigma_eta=0.1,
    theta=0.1,
    mu=100.0,
    x0=10.0,
    random_state=None,
):
    """
    Generates two cointegrated time series X and Y with nonstationary behavior.

    Args:
        N (int): Number of time steps.
        sigma_x (float): Std deviation of X's random walk noise.
        sigma_eta (float): Noise amplitude of epsilon.
        theta (float): Friction parameter, speed of mean reversion of epsilon.
        mu (float): Long-run mean of the error process.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple: Arrays (X, Y) of shape (N,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X, epsilon = np.zeros(N), np.zeros(N)

    # Initial conditions
    X[0] = np.random.normal(x0, sigma_x)
    epsilon[0] = mu + np.random.normal(
        0.0, sigma_eta
    )  # Random initial value for epsilon

    for t in range(1, N):
        if t % 200 == 0:
            # model slow evolution of equilibrium and very rare unexpected
            # large shifts.
            mu += np.random.standard_cauchy() * 0.01

        if np.random.uniform(0, 1) >= 0.995:
            mu += np.random.uniform(
                -3, 3
            )  # model rare discrete regime changes or market discontinuities.

        X[t] = X[t - 1] + np.random.normal(0.0, sigma_x)  # Random walk
        # Euler–Maruyama step for epsilon: ε_t = ε_{t-1} + θ(μ - ε_{t-1}) + η_t,   η_t∼𝒩(0,σ_{η}^2)
        # Assume Δt = 1 for simplicity
        epsilon[t] = (
            epsilon[t - 1]
            + theta * (mu - epsilon[t - 1])
            + np.random.normal(0.0, sigma_eta)
        )

    return X, X + epsilon  # Y = X + ε


def make_ohlcv(
    close,
    window=24,
    stride=24,
    base_volume=1000,
    return_sensitivity=1000,
    volatility_sensitivity=1000,
    random_state=None,
):
    """
    Create OHLC and volume data from a close price series using numpy.

    Volume is modeled as a function of price volatility and absolute return magnitude.

    Args:
        close (np.ndarray): Close price series as a numpy array.
        window (int): Window size for OHLC calculation.
        stride (int): Stride size for moving window.
        base_volume (float): Base trading volume level.
        volatility_sensitivity (float): Multiplier for volume response to price dynamics.
        random_state (int or None): For reproducibility.

    Returns:
        tuple: Tuple of numpy arrays (open, high, low, close, volume).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(close)
    n_windows = (n - window) // stride + 1

    open_ = np.zeros(n_windows)
    high_ = np.zeros(n_windows)
    low_ = np.zeros(n_windows)
    close_ = np.zeros(n_windows)
    volume_ = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * stride
        end = start + window
        segment = close[start:end]

        open_[i] = segment[0]
        high_[i] = segment.max()
        low_[i] = segment.min()
        close_[i] = segment[-1]

        # Calculate volatility
        returns = np.diff(segment) / segment[:-1]
        volatility = np.std(returns)
        volume_[i] = int(base_volume * return_sensitivity * abs(returns).sum() * volatility_sensitivity * volatility)

    return open_, high_, low_, close_, volume_


def excess_return_metric(history):
    pv = history["portfolio_valuation"]       # array of portfolio values
    price = history["data_close"]             # array of prices
    # buy-and-hold market value
    market = pv[0] * (price / price[0])       
    # compute period returns
    ret_p = pv[1:] / pv[:-1] - 1
    ret_m = market[1:] / market[:-1] - 1
    # excess return series
    excess = ret_p - ret_m
    # return cumulative excess over the episode
    return np.nansum(excess)

def extract_signals(historical_info_df: pd.DataFrame):
    """
    Given a DataFrame with columns ['data_close', 'position'], 
    shift and detect only the entry/exit transitions.
    Returns (buy_df, sell_df, exit_df).
    """
    df = historical_info_df[['data_close', 'position']].copy()
    df['position'] = df['position'].shift(-1)
    df['prev_position'] = df['position'].shift(1)

    buy_df = df[(df['position'] == 1) & (df['prev_position'] != 1)]
    sell_df = df[(df['position'] == -1) & (df['prev_position'] != -1)]
    exit_df = df[(df['position'] == 0) & (df['prev_position'] != 0)]

    return buy_df, sell_df, exit_df