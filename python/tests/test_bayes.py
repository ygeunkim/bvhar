import pytest
from bvhar.model import VarBayes, LdltConfig, InterceptConfig, SsvsConfig, HorseshoeConfig
import numpy as np

def test_var_bayes():
    num_data = 30
    dim_data = 3
    var_lag = 2
    data = np.random.randn(num_data, dim_data)

    num_chains = 2
    num_threads = 2
    num_iter = 5
    num_burn = 2
    thin = 1
    intercept = True
    minnesota = True

    np.random.seed(1)
    fit_var_ssvs = VarBayes(
        data, var_lag,
        num_chains,
        num_iter,
        num_burn,
        thin,
        SsvsConfig(),
        LdltConfig(),
        InterceptConfig(),
        intercept,
        minnesota,
        False,
        num_threads
    )
    fit_var_ssvs.fit()

    np.random.seed(1)
    fit_var_hs = VarBayes(
        data, var_lag,
        num_chains,
        num_iter,
        num_burn,
        thin,
        HorseshoeConfig(),
        LdltConfig(),
        InterceptConfig(),
        intercept,
        minnesota,
        False,
        num_threads
    )
    fit_var_hs.fit()

    assert fit_var_ssvs.n_features_in_ == dim_data
    assert fit_var_ssvs.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_var_ssvs.intercept_.shape == (dim_data,)

    assert fit_var_hs.n_features_in_ == dim_data
    assert fit_var_hs.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_var_hs.intercept_.shape == (dim_data,)

    with pytest.warns(UserWarning, match=f"'n_thread = 3 > 'n_chain' = {num_chains}' will not use every thread. Specify as 'n_thread <= 'n_chain'."):
        VarBayes(
            data, var_lag, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, 3
        )
    
    with pytest.raises(ValueError, match=f"'data' rows must be larger than `lag` = {var_lag}"):
        data = np.random.randn(var_lag - 1, dim_data)
        VarBayes(
            data, var_lag, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, num_threads
        )
