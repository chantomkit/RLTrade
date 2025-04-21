import numpy as np
import pandas as pd

class FeatureEngineering:
    """
    A flexible feature-engineering pipeline for time series DataFrame.
    Build a sequence of feature transformations via a configuration list.
    """
    def __init__(self, df, feature_prefix='feature_'):
        self.df = df.copy()
        self.prefix = feature_prefix
        self._builders = {
            'select': self._select_features,
            'rolling': self._rolling_window,
            'rolling_mean_corrected': self._rolling_mean_corrected
        }

    @staticmethod
    def _transform(series, mode):
        """
        Apply a transformation to a pandas Series.
        Supported modes: 'default', 'log', 'diff', 'logdiff', 'pct_change', 'logpct_change'
        """
        if mode == 'default':
            return series
        if mode == 'log':
            return np.log(series)
        if mode == 'diff':
            return series.diff().fillna(0)
        if mode == 'logdiff':
            return np.log(series).diff().fillna(0)
        if mode == 'pct_change':
            return series.pct_change().fillna(0)
        if mode == 'logpct_change':
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
            ftype = step['type']
            params = step.get('params', {})
            builder = self._builders.get(ftype)
            if not builder:
                raise ValueError(f"Unknown feature type: {ftype}")
            df = builder(df, **params)
        self.df = df
        return df

    def _select_features(self, df, cols):
        """Select original columns and add them with feature prefix."""
        selected = df[cols].add_prefix(self.prefix)
        return pd.concat([df, selected], axis=1)

    def _rolling_window(self, df, cols, window=24, mode='default', subtract_mean=False):
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

    def _rolling_mean_corrected(self, df, cols, window=24, mode='default'):
        """Subtract rolling mean from transformed series for given columns."""
        result = df.copy()
        for col in cols:
            series = self._transform(df[col], mode)
            roll_mean = series.rolling(window=window, min_periods=1).mean()
            corrected = series - roll_mean
            result[f"{self.prefix}{col}_rolling_mean_corrected"] = corrected
        return result

def stationary_dgp(
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

    X, epsilon = np.zeros(N), np.zeros(N)  # +1 because we use epsilon[0] as initial value

    # Initial conditions
    X[0] = np.random.normal(x0, sigma_x)
    epsilon[0] = mu + np.random.normal(0.0, sigma_eta)  # Random initial value for epsilon

    for t in range(1, N):
        X[t] = X[t - 1] + np.random.normal(0.0, sigma_x)  # Random walk

        # Euler–Maruyama step for epsilon: ε_t = ε_{t-1} + θ(μ - ε_{t-1}) + η_t,   η_t∼𝒩(0,σ_{η}^2)
        # Assume Δt = 1 for simplicity
        epsilon[t] = (
            epsilon[t - 1] +
            theta * (mu - epsilon[t - 1]) +
            np.random.normal(0.0, sigma_eta)
        )

    return X, X + epsilon  # Y = X + ε

def nonstationary_dgp(
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

    X, epsilon = np.zeros(N), np.zeros(N)  # +1 because we use epsilon[0] as initial value

    # Initial conditions
    X[0] = np.random.normal(x0, sigma_x)
    epsilon[0] = mu + np.random.normal(0.0, sigma_eta)  # Random initial value for epsilon

    for t in range(1, N):
        if t % 200 == 0:
            mu += np.random.standard_cauchy() * 0.01  # model slow evolution of equilibrium and very rare unexpected large shifts.

        if np.random.uniform(0, 1) >= 0.995:
            mu += np.random.uniform(-3, 3)  # model rare discrete regime changes or market discontinuities.

        X[t] = X[t - 1] + np.random.normal(0.0, sigma_x)  # Random walk
        # Euler–Maruyama step for epsilon: ε_t = ε_{t-1} + θ(μ - ε_{t-1}) + η_t,   η_t∼𝒩(0,σ_{η}^2)
        # Assume Δt = 1 for simplicity
        epsilon[t] = (
            epsilon[t - 1] +
            theta * (mu - epsilon[t - 1]) +
            np.random.normal(0.0, sigma_eta)
        )

    return X, X + epsilon  # Y = X + ε