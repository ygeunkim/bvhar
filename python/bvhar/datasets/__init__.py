import os
import pickle
import pkg_resources

_data_path = pkg_resources.resource_filename(__name__, 'data/etf_vix.pkl')

with open(_data_path, 'rb') as f:
    etf_vix = pickle.load(f)
