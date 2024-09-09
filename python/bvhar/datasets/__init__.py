import sys
import pandas as pd
if sys.version_info >= (3, 10):
    from importlib.resources import files
    _data_path = files('bvhar.datasets.data') / 'etf_vix.csv'
else:
    from importlib.resources import path
    with resources.path('bvhar.datasets.data', 'etf_vix.csv') as _data_path:
        _data_path = str(_data_path)

etf_vix = pd.read_csv(_data_path)

__all__ = [
    "etf_vix"
]