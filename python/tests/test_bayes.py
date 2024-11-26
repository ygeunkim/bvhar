import pytest
from bvhar.model import VarBayes, VharBayes
from bvhar.model import LdltConfig, SvConfig, InterceptConfig
from bvhar.model import SsvsConfig, HorseshoeConfig, MinnesotaConfig, LambdaConfig, NgConfig, DlConfig, GdpConfig
from bvhar.datasets import load_vix
import numpy as np

def help_var_bayes(
    dim_data, var_lag, data,
    num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
    bayes_config, cov_config,
    test_y = None, n_ahead = None, pred = False, roll = False, expand = False, spillover = False
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
        ggl,
        False,
        num_threads
    )
    fit_bayes.fit()

    assert fit_bayes.n_features_in_ == dim_data
    assert fit_bayes.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_bayes.intercept_.shape == (dim_data,)

    if pred:
        pred_out = fit_bayes.predict(n_ahead, stable = False, sparse = True)
        assert pred_out['forecast'].shape == (n_ahead, dim_data)
        assert pred_out['se'].shape == (n_ahead, dim_data)
        assert pred_out['lower'].shape == (n_ahead, dim_data)
        assert pred_out['upper'].shape == (n_ahead, dim_data)
    if roll:
        roll_out = fit_bayes.roll_forecast(1, test_y, stable = False, sparse = True)
        assert roll_out['forecast'].shape == (n_ahead, dim_data)
        assert roll_out['se'].shape == (n_ahead, dim_data)
        assert roll_out['lower'].shape == (n_ahead, dim_data)
        assert roll_out['upper'].shape == (n_ahead, dim_data)
    if expand:
        roll_out = fit_bayes.expand_forecast(1, test_y, stable = False, sparse = True)
        assert roll_out['forecast'].shape == (n_ahead, dim_data)
        assert roll_out['se'].shape == (n_ahead, dim_data)
        assert roll_out['lower'].shape == (n_ahead, dim_data)
        assert roll_out['upper'].shape == (n_ahead, dim_data)

def test_var_bayes():
    num_data = 50
    dim_data = 2
    var_lag = 3
    etf_vix = load_vix()
    data = etf_vix.to_numpy()[:num_data, :dim_data]
    n_ahead = 5
    data_out = etf_vix.to_numpy()[num_data:(num_data + n_ahead), :dim_data]

    num_chains = 2
    num_threads = 1
    num_iter = 5
    num_burn = 2
    thin = 1
    intercept = True
    minnesota = True
    ggl = True

    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        SsvsConfig(), LdltConfig(),
        data_out, n_ahead, True, True, True
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        HorseshoeConfig(), LdltConfig(),
        data_out, n_ahead
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        MinnesotaConfig(lam=LambdaConfig()), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        NgConfig(), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        DlConfig(), LdltConfig()
    )
    help_var_bayes(
        dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        GdpConfig(), LdltConfig()
    )

    # help_var_bayes(
    #     dim_data, var_lag, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
    #     SsvsConfig(), SvConfig()
    # )

    with pytest.warns(UserWarning, match=f"'n_thread = 3 > 'n_chain' = 2' will not use every thread. Specify as 'n_thread <= 'n_chain'."):
        VarBayes(
            data, var_lag, 2, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, True, 3
        )
    
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'lag' = {var_lag}"):
        etf_vix = load_vix()
        data = etf_vix.iloc[:(var_lag - 1), :dim_data]
        VarBayes(
            data, var_lag, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, True, num_threads
        )

def help_vhar_bayes(
    dim_data, week, month, data,
    num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
    bayes_config, cov_config,
    test_y = None, n_ahead = None, pred = False, roll = False, expand = False, spillover = False
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
        ggl,
        False,
        num_threads
    )
    fit_bayes.fit()

    assert fit_bayes.n_features_in_ == dim_data
    assert fit_bayes.coef_.shape == (dim_data * 3 + 1, dim_data)
    assert fit_bayes.intercept_.shape == (dim_data,)

    if pred:
        pred_out = fit_bayes.predict(n_ahead, stable = False, sparse = True)
        assert pred_out['forecast'].shape == (n_ahead, dim_data)
        assert pred_out['se'].shape == (n_ahead, dim_data)
        assert pred_out['lower'].shape == (n_ahead, dim_data)
        assert pred_out['upper'].shape == (n_ahead, dim_data)
    if roll:
        roll_out = fit_bayes.roll_forecast(1, test_y, stable = False, sparse = True)
        assert roll_out['forecast'].shape == (n_ahead, dim_data)
        assert roll_out['se'].shape == (n_ahead, dim_data)
        assert roll_out['lower'].shape == (n_ahead, dim_data)
        assert roll_out['upper'].shape == (n_ahead, dim_data)
    if expand:
        roll_out = fit_bayes.expand_forecast(1, test_y, stable = False, sparse = True)
        assert roll_out['forecast'].shape == (n_ahead, dim_data)
        assert roll_out['se'].shape == (n_ahead, dim_data)
        assert roll_out['lower'].shape == (n_ahead, dim_data)
        assert roll_out['upper'].shape == (n_ahead, dim_data)

def test_vhar_bayes():
    num_data = 50
    dim_data = 3
    week = 5
    month = 22
    etf_vix = load_vix()
    data = etf_vix.iloc[:num_data, :dim_data]
    n_ahead = 5
    data_out = etf_vix.iloc[num_data:(num_data + n_ahead), :dim_data]

    num_chains = 2
    num_threads = 1
    num_iter = 5
    num_burn = 2
    thin = 1
    intercept = True
    minnesota = "longrun"
    ggl = True

    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        SsvsConfig(), LdltConfig(),
        data_out, n_ahead, True, True, True
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        HorseshoeConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        MinnesotaConfig(lam=LambdaConfig()), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        NgConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        DlConfig(), LdltConfig()
    )
    help_vhar_bayes(
        dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota, ggl,
        GdpConfig(), LdltConfig()
    )

    # help_vhar_bayes(
    #     dim_data, week, month, data, num_chains, num_threads, num_iter, num_burn, thin, intercept, minnesota,
    #     SsvsConfig(), SvConfig()
    # )

    with pytest.warns(UserWarning, match=f"'n_thread = 3 > 'n_chain' = 2' will not use every thread. Specify as 'n_thread <= 'n_chain'."):
        VharBayes(
            data, week, month, 2, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, True, 3
        )
    
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'lag' = {month}"):
        data = etf_vix.iloc[:(month - 1), :dim_data]
        VharBayes(
            data, week, month, num_chains, num_iter, num_burn, thin,
            SsvsConfig(), LdltConfig(), InterceptConfig(),
            intercept, minnesota, False, True, num_threads
        )