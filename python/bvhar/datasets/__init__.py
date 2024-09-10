import sys
import pandas as pd
if sys.version_info >= (3, 10):
    from importlib.resources import files
    _data_path = files('bvhar.datasets.data') / 'etf_vix.csv'
else:
    from importlib.resources import path
    with path('bvhar.datasets.data', 'etf_vix.csv') as _data_path:
        _data_path = str(_data_path)

def load_vix():
    """Load and return the CBOE VIX of several ETF datasets

    Returns
    -------
    pd.DataFrame
        Dataframe for CBOE VIX series
    """
    return pd.read_csv(_data_path)

# etf_vix = pd.read_csv(_data_path)

__all__ = [
    "load_vix"
]