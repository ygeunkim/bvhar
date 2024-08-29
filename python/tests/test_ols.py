# import pytest
# from bvhar.model import OlsVar, OlsVhar
from bvhar.model import VarOls, VharOls
import numpy as np

def test_var():
    num_data = 30
    dim_data = 3
    var_lag = 2
    data = np.random.randn(num_data, dim_data)
    # fit_var = OlsVar(data, var_lag, True, 1)
    # fit_var_llt = OlsVar(data, var_lag, True, 2)
    # fit_var_qr = OlsVar(data, var_lag, True, 3)
    # res = fit_var.return_ols_res()
    # res_llt = fit_var_llt.return_ols_res()
    # res_qr = fit_var_qr.return_ols_res()
    # res_keys = ['coefficients', 'fitted.values', 'residuals', 'covmat', 'df', 'm', 'obs', 'y0',
    #              'p', 'totobs', 'process', 'type', 'design', 'y']
    # assert set(res_keys).issubset(set(res.keys()))
    # assert set(res_keys).issubset(set(res_llt.keys()))
    # assert set(res_keys).issubset(set(res_qr.keys()))
    fit_var = VarOls(data, var_lag, True, "nor")
    fit_var_llt = VarOls(data, var_lag, True, "chol")
    fit_var_qr = VarOls(data, var_lag, True, "qr")
    fit_var.fit()
    fit_var_llt.fit()
    fit_var_qr.fit()
    

def test_vhar():
    num_data = 30
    dim_data = 3
    week = 5
    month = 22
    data = np.random.randn(num_data, dim_data)
    # fit_vhar = OlsVhar(data, week, month, True, 1)
    # fit_vhar_llt = OlsVhar(data, week, month, True, 2)
    # fit_vhar_qr = OlsVhar(data, week, month, True, 3)
    # res = fit_vhar.return_ols_res()
    # res_llt = fit_vhar_llt.return_ols_res()
    # res_qr = fit_vhar_qr.return_ols_res()
    # res_keys = ['coefficients', 'fitted.values', 'residuals', 'covmat', 'df', 'm', 'obs', 'y0',
    #              'p', 'week', 'month', 'totobs', 'process', 'type', 'HARtrans', 'design', 'y']
    # assert set(res_keys).issubset(set(res.keys()))
    # assert set(res_keys).issubset(set(res_llt.keys()))
    # assert set(res_keys).issubset(set(res_qr.keys()))
    fit_vhar = VharOls(data, 5, 22, True, "nor")
    fit_vhar_llt = VharOls(data, 5, 22, True, "chol")
    fit_vhar_qr = VharOls(data, 5, 22, True, "qr")
    fit_vhar.fit()
    fit_vhar_llt.fit()
    fit_vhar_qr.fit()