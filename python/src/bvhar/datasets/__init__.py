import pandas as pd
from importlib.resources import files

def load_vix():
    """Load and return the CBOE VIX of several ETF datasets

    Returns
    -------
    pd.DataFrame
        Dataframe for CBOE VIX series
    """
    _data_path = files('bvhar.datasets.data') / 'etf_vix.csv'
    return pd.read_csv(_data_path)

__all__ = [
    "load_vix"
]