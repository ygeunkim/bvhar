import pytest
from bvhar.model import VarOls, VharOls
import numpy as np

def test_var():
    num_data = 30
    dim_data = 3
    var_lag = 2
    data = np.random.randn(num_data, dim_data)

    fit_var = VarOls(data, var_lag, True, "nor")
    fit_var_llt = VarOls(data, var_lag, True, "chol")
    fit_var_qr = VarOls(data, var_lag, True, "qr")
    fit_var.fit()
    fit_var_llt.fit()
    fit_var_qr.fit()

    assert fit_var.n_features_in_ == dim_data
    assert fit_var.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_var.intercept_.shape == (dim_data,)

    data = np.random.randn(var_lag - 1, dim_data)
    with pytest.raises(ValueError, match=f"'data' rows must be larger than `lag` = {var_lag}"):
        fit_var = VarOls(data, var_lag, True, "nor")

def test_vhar():
    num_data = 30
    dim_data = 3
    week = 5
    month = 22
    data = np.random.randn(num_data, dim_data)

    fit_vhar = VharOls(data, week, month, True, "nor")
    fit_vhar_llt = VharOls(data, week, month, True, "chol")
    fit_vhar_qr = VharOls(data, week, month, True, "qr")
    fit_vhar.fit()
    fit_vhar_llt.fit()
    fit_vhar_qr.fit()

    assert fit_vhar.n_features_in_ == dim_data
    assert fit_vhar.coef_.shape == (dim_data * 3 + 1, dim_data)
    assert fit_vhar.intercept_.shape == (dim_data,)

    data = np.random.randn(month - 1, dim_data)
    with pytest.raises(ValueError, match=f"'data' rows must be larger than `lag` = {month}"):
        fit_vhar = VharOls(data, week, month, True, "nor")