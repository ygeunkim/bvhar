from importlib import resources
import pandas as pd

_data_path = resources.files('bvhar.datasets.data') / 'etf_vix.csv'
etf_vix = pd.read_csv(_data_path)

__all__ = [
    "etf_vix"
]