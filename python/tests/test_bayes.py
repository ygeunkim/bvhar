import pytest
from bvhar.model import VarBayes, VharBayes
from bvhar.model import LdltConfig, SvConfig, InterceptConfig
from bvhar.model import SsvsConfig, HorseshoeConfig, MinnesotaConfig, LambdaConfig, NgConfig, DlConfig
import numpy as np

def help_var_bayes(
    dim_data, var_lag, data,
    num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
    bayes_config, cov_config
):
    np.random.seed(1)
    fit_bayes = VarBayes(
        data, var_lag,
        num_chains,
        num_iter,
        num_burn,
        thin,
        bayes_config,
        cov_config,
        InterceptConfig(),
        intercept,
        minnesota,
        False,
        num_threads
    )
    fit_bayes.fit()

    assert fit_bayes.n_features_in_ == dim_data
    assert fit_bayes.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_bayes.intercept_.shape == (dim_data,)

def test_var_bayes():
    num_data = 30
    dim_data = 2
    var_lag = 3

    np.random.seed(1)
    data = np.random.randn(num_data, dim_data)

    num_chains = 2
    num_threads = 1
    num_iter = 5
    num_burn = 2
    thin = 1
    intercept = True
    minnesota = True

    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        SsvsConfig(), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        HorseshoeConfig(), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        MinnesotaConfig(lam=LambdaConfig()), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        NgConfig(), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        DlConfig(), LdltConfig()
    )

    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        SsvsConfig(), SvConfig()
    )

    with pytest.warns(UserWarning, match=f"'n_thread = 3 > 'n_chain' = {num_chains}' will not use every thread. Specify as 'n_thread <= 'n_chain'."):
        VarBayes(
            data, var_lag, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, 3
        )
    
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'lag' = {var_lag}"):
        data = np.random.randn(var_lag - 1, dim_data)
        VarBayes(
            data, var_lag, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, num_threads
        )

def help_vhar_bayes(
    dim_data, week, month, data,
    num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
    bayes_config, cov_config
):
    np.random.seed(1)
    fit_bayes = VharBayes(
        data, week, month,
        num_chains,
        num_iter,
        num_burn,
        thin,
        bayes_config,
        cov_config,
        InterceptConfig(),
        intercept,
        minnesota,
        False,
        num_threads
    )
    fit_bayes.fit()

    assert fit_bayes.n_features_in_ == dim_data
    assert fit_bayes.coef_.shape == (dim_data * 3 + 1, dim_data)
    assert fit_bayes.intercept_.shape == (dim_data,)

def test_vhar_bayes():
    num_data = 30
    dim_data = 3
    week = 5
    month = 22

    np.random.seed(1)
    data = np.random.randn(num_data, dim_data)

    num_chains = 2
    num_threads = 2
    num_iter = 5
    num_burn = 2
    thin = 1
    intercept = True
    minnesota = "longrun"

    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        SsvsConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        HorseshoeConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        MinnesotaConfig(lam=LambdaConfig()), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        NgConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        DlConfig(), LdltConfig()
    )

    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
        SsvsConfig(), SvConfig()
    )

    with pytest.warns(UserWarning, match=f"'n_thread = 3 > 'n_chain' = {num_chains}' will not use every thread. Specify as 'n_thread <= 'n_chain'."):
        VharBayes(
            data, week, month, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, 3
        )
    
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'lag' = {month}"):
        data = np.random.randn(month - 1, dim_data)
        VharBayes(
            data, week, month, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, num_threads
        )