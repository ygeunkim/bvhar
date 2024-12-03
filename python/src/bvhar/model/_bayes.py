from ..utils._misc import make_fortran_array, check_np, build_grpmat, process_record, concat_chain, concat_params, process_dens_forecast
from ..utils.checkomp import get_maxomp
from .._src._design import build_response, build_design
from .._src._ldlt import McmcLdlt, McmcLdltGrp
from .._src._ldltforecast import LdltForecast, LdltVarRoll, LdltVharRoll, LdltVarExpand, LdltVharExpand, LdltGrpVarRoll, LdltGrpVharRoll, LdltGrpVarExpand, LdltGrpVharExpand
from .._src._sv import SvMcmc, SvGrpMcmc
from .._src._svforecast import SvForecast, SvVarRoll, SvVharRoll, SvVarExpand, SvVharExpand, SvGrpVarRoll, SvGrpVharRoll, SvGrpVarExpand, SvGrpVharExpand
from ._spec import LdltConfig, SvConfig, InterceptConfig
from ._spec import _BayesConfig, SsvsConfig, HorseshoeConfig, MinnesotaConfig, DlConfig, NgConfig, GdpConfig
import numpy as np
import pandas as pd
import warnings
from math import floor

class _AutoregBayes:
    """Base class for Bayesian estimation"""
    def __init__(
        self, data, lag, p, n_chain = 1, n_iter = 1000,
        n_burn = None, n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(), fit_intercept = True,
        minnesota = "longrun", ggl = True
    ):
        self.y_ = check_np(data)
        self.n_features_in_ = self.y_.shape[1]
        self.p_ = p # 3 in VHAR
        self.lag_ = lag # month in VHAR
        if self.y_.shape[0] <= self.lag_:
            raise ValueError(f"'data' rows must be larger than 'lag' = {self.lag_}")
        # self.design_ = build_design(self.y, self.lag_, fit_intercept)
        # self.response_ = build_response(self.y, self.lag_, self.lag_ + 1)
        self.chains_ = int(n_chain)
        # self.thread_ = n_thread
        self.iter_ = int(n_iter)
        if n_burn is None:
            n_burn = floor(n_iter / 2)
        self.burn_ = int(n_burn)
        self.thin_ = int(n_thin)
        self.fit_intercept = fit_intercept
        self.group_ = build_grpmat(self.p_, self.n_features_in_, minnesota)
        self._group_id = pd.unique(self.group_.flatten(order='F')).astype(np.int32)
        self._own_id = None
        self._cross_id = None
        self._ggl = ggl
        n_grp = len(self._group_id)
        n_alpha = self.n_features_in_ * self.n_features_in_ * self.p_
        n_design = self.p_ * self.n_features_in_ + 1 if self.fit_intercept else self.p_ * self.n_features_in_
        n_eta = int(self.n_features_in_ * (self.n_features_in_ - 1) / 2)
        self.cov_spec_ = cov_config
        self.spec_ = bayes_config
        self.intercept_spec_ = intercept_config
        self.init_ = [
            {
                'init_coef': np.random.uniform(-1, 1, (n_design, self.n_features_in_)),
                'init_contem': np.exp(np.random.uniform(-1, 0, n_eta))
            }
            for _ in range(self.chains_)
        ]
        if type(self.cov_spec_) == LdltConfig:
            for init in self.init_:
                init.update({
                    'init_diag': np.exp(np.random.uniform(-1, 1, self.n_features_in_))
                })
        elif type(self.cov_spec_) == SvConfig:
            for init in self.init_:
                init.update({
                    'lvol_init': np.random.uniform(-1, 1, self.n_features_in_),
                    'lvol': np.exp(np.random.uniform(-1, 1, self.n_features_in_ * n_design)).reshape(self.n_features_in_, -1).T,
                    'lvol_sig': [np.exp(np.random.uniform(-1, 1))]
                })
        if type(self.spec_) == SsvsConfig:
            for init in self.init_:
                coef_mixture = np.random.uniform(-1, 1, n_grp)
                coef_mixture = np.exp(coef_mixture) / (1 + np.exp(coef_mixture))
                init_coef_dummy = np.random.binomial(1, 0.5, n_alpha)
                chol_mixture = np.random.uniform(-1, 1, n_eta)
                chol_mixture = np.exp(chol_mixture) / (1 + np.exp(chol_mixture))
                init_coef_slab = np.exp(np.random.uniform(-1, 1, n_alpha))
                init_contem_slab = np.exp(np.random.uniform(-1, 1, n_eta))
                init.update({
                    'init_coef_dummy': init_coef_dummy,
                    'coef_mixture': coef_mixture,
                    'coef_slab': init_coef_slab,
                    'chol_mixture': chol_mixture,
                    'contem_slab': init_contem_slab,
                    'coef_spike_scl': np.random.uniform(0, 1),
                    'chol_spike_scl': np.random.uniform(0, 1)
                })
        elif type(self.spec_) == HorseshoeConfig:
            for init in self.init_:
                local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
                global_sparsity = np.exp(np.random.uniform(-1, 1))
                group_sparsity = np.exp(np.random.uniform(-1, 1, n_grp))
                contem_local_sparsity = np.exp(np.random.uniform(-1, 1, n_eta))
                contem_global_sparsity = np.exp(np.random.uniform(-1, 1))
                init.update({
                    'local_sparsity': local_sparsity,
                    'global_sparsity': global_sparsity,
                    'group_sparsity': group_sparsity,
                    'contem_local_sparsity': contem_local_sparsity,
                    'contem_global_sparsity': np.array([contem_global_sparsity]) # used as VectorXd in C++
                })
        elif type(self.spec_) == MinnesotaConfig:
            for init in self.init_:
                init.update({
                    'own_lambda': np.random.uniform(0, 1),
                    'cross_lambda': np.random.uniform(0, 1),
                    'contem_lambda': np.random.uniform(0, 1)
                })
        elif type(self.spec_) == DlConfig:
            for init in self.init_:
                local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
                global_sparsity = np.exp(np.random.uniform(-1, 1))
                contem_local_sparsity = np.exp(np.random.uniform(-1, 1, n_eta))
                contem_global_sparsity = np.exp(np.random.uniform(-1, 1))
                init.update({
                    'local_sparsity': local_sparsity,
                    'global_sparsity': global_sparsity,
                    'contem_local_sparsity': contem_local_sparsity,
                    'contem_global_sparsity': np.array([contem_global_sparsity]) # used as VectorXd in C++
                })
        elif type(self.spec_) == NgConfig:
            for init in self.init_:
                local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
                global_sparsity = np.exp(np.random.uniform(-1, 1))
                group_sparsity = np.exp(np.random.uniform(-1, 1, n_grp))
                contem_local_sparsity = np.exp(np.random.uniform(-1, 1, n_eta))
                contem_global_sparsity = np.exp(np.random.uniform(-1, 1))
                local_shape = np.random.uniform(0, 1, n_grp)
                contem_shape = np.random.uniform(0, 1)
                init.update({
                    'local_shape': local_shape,
                    'contem_shape': contem_shape,
                    'local_sparsity': local_sparsity,
                    'global_sparsity': global_sparsity,
                    'group_sparsity': group_sparsity,
                    'contem_local_sparsity': contem_local_sparsity,
                    'contem_global_sparsity': np.array([contem_global_sparsity]) # used as VectorXd in C++
                })
        elif type(self.spec_) == GdpConfig:
            for init in self.init_:
                local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
                group_rate = np.exp(np.random.uniform(-1, 1, n_grp))
                contem_local_sparsity = np.exp(np.random.uniform(-1, 1, n_eta))
                contem_local_rate = np.exp(np.random.uniform(-1, 1, n_eta))
                coef_shape = np.random.uniform(0, 1)
                coef_rate = np.random.uniform(0, 1)
                contem_shape = np.random.uniform(0, 1)
                contem_rate = np.random.uniform(0, 1)
                init.update({
                    'local_sparsity': local_sparsity,
                    'group_rate': group_rate,
                    'contem_local_sparsity': contem_local_sparsity,
                    'contem_rate': contem_local_rate,
                    'gamma_shape': coef_shape,
                    'gamma_rate': coef_rate,
                    'contem_gamma_shape': contem_shape,
                    'contem_gamma_rate': contem_rate
                })
        self.init_ = make_fortran_array(self.init_)
        self._prior_type = {
            "Minnesota": 1,
            "SSVS": 2,
            "Horseshoe": 3,
            "HMN": 4,
            "NG": 5,
            "DL": 6,
            "GDP": 7
        }.get(self.spec_.prior)
        self.is_fitted_ = False
        self.coef_ = None
        self.intercept_ = None
        self.param_names_ = None
        self.param_ = None
        # self.cov_ = None
        self.sparse_coef_ = None
    
    def _validate(self):
        if not isinstance(self.cov_spec_, LdltConfig):
            raise TypeError("`cov_config` should be `LdltConfig` or `SvConfig`.")
        if not isinstance(self.intercept_spec_, InterceptConfig):
            raise TypeError("`intercept_config` should be `InterceptConfig` when 'fit_intercept' is True.")
        if not isinstance(self.spec_, _BayesConfig):
            raise TypeError("`bayes_spec` should be the derived class of `_BayesConfig`.")
        self.cov_spec_.update(self.n_features_in_)
        self.intercept_spec_.update(self.n_features_in_)
        if type(self.spec_) == SsvsConfig:
            self.spec_.update(self._group_id, self._own_id, self._cross_id)
        elif type(self.spec_) == HorseshoeConfig:
            pass
        elif type(self.spec_) == MinnesotaConfig:
            self.spec_.update(self.y_, self.p_, self.n_features_in_)
        elif type(self.spec_) == DlConfig:
            pass
        elif type(self.spec_) == NgConfig:
            pass
        elif type(self.spec_) == GdpConfig:
            pass

    def fit(self):
        pass

    def predict(self):
        pass

    def roll_forecast(self):
        pass

    def expand_forecast(self):
        pass

    def spillover(self):
        pass

    def dynamic_spillover(self):
        pass

class VarBayes(_AutoregBayes):
    """Bayesian Vector Autoregressive Model

    Fits Bayesian VAR model.

    Parameters
    ----------
    data : array-like
        Time series data of which columns indicate the variables
    lag : int
        VAR lag, by default 1
    n_chain : int
        Number of MCMC chains, by default 1
    n_iter : int
        Number of MCMC total iterations, by default 1000
    n_burn : int
        MCMC burn-in (warm-up), by default `floor(n_iter / 2)`
    n_thin : int
        Thinning every `n_thin`-th iteration, by default 1
    bayes_config : '_BayesConfig'
        Prior configuration, by default SsvsConfig()
    cov_config : {'LdltConfig', 'SvConfig'}
        Prior configuration for covariance matrix, by default LdltConfig()
    intercept_config : 'InterceptConfig'
        Prior configuration for constant term, by default InterceptConfig()
    fit_intercept : bool
        Include constant term in the model, by default True
    minnesota : bool
        If `True`, apply Minnesota-type group structure, by default True
    ggl : bool
        If `False`, use group shrinkage parameter instead of global, by default True
    verbose : bool
        If `True`, print progress bar for MCMC, by default False
    n_thread : int
        Number of OpenMP threads, by default 1
    
    Attributes
    ----------
    coef_ : ndarray
        VHAR coefficient matrix.
    intercept_ : ndarray
        VHAR model constant vector.
    n_features_in_ : int
        Number of variables.
    """
    def __init__(
        self,
        data,
        lag = 1,
        n_chain = 1,
        n_iter = 1000,
        n_burn = None,
        n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(),
        fit_intercept = True,
        minnesota = True,
        ggl = True,
        verbose = False,
        n_thread = 1
    ):
        super().__init__(data, lag, lag, n_chain, n_iter, n_burn, n_thin, bayes_config, cov_config, intercept_config, fit_intercept, "short" if minnesota else "no", ggl)
        self.design_ = build_design(self.y_, lag, fit_intercept)
        self.response_ = build_response(self.y_, lag, lag + 1)
        if minnesota:
            self._own_id = np.array([2], dtype=np.int32)
            self._cross_id = np.arange(1, self.p_ + 2, dtype=np.int32)
            self._cross_id = np.delete(self._cross_id, 1)
        else:
            self._own_id = np.array([2], dtype=np.int32)
            self._cross_id = np.array([2], dtype=np.int32)
        self._validate()
        self.thread_ = n_thread
        if self.thread_ > get_maxomp():
            warnings.warn(f"'n_thread' = {self.thread_} is greather than 'omp_get_max_threads()' = {get_maxomp()}. Check with utils.checkomp.get_maxomp(). Check OpenMP support of your machine with utils.checkomp.is_omp().")
        if self.thread_ > n_chain and n_chain != 1:
            warnings.warn(f"'n_thread = {self.thread_} > 'n_chain' = {n_chain}' will not use every thread. Specify as 'n_thread <= 'n_chain'.")
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                self.__model = McmcLdlt(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
            else:
                self.__model = McmcLdltGrp(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
        else:
            if self._ggl:
                self.__model = SvMcmc(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
            else:
                self.__model = SvGrpMcmc(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )

    def fit(self):
        """Conduct MCMC and compute posterior mean
        Returns
        -------
        self : object
            An instance of the estimator.
        """
        res = self.__model.returnRecords()
        self.param_names_ = process_record(res)
        self.param_ = concat_chain(res)
        self.coef_ = self.param_.filter(regex='^alpha\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T
        self.sparse_coef_ = self.param_.filter(regex='^alpha_sparse\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T
        if self.fit_intercept:
            self.intercept_ = self.param_.filter(regex='^c\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T
            self.coef_ = np.concatenate([self.coef_, self.intercept_], axis=0)
            self.sparse_coef_ = np.concatenate([self.sparse_coef_, self.param_.filter(regex='^c_sparse\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T], axis=0)
            self.intercept_ = self.intercept_.reshape(self.n_features_in_,)
        self.is_fitted_ = True

    def predict(self, n_ahead: int, level = .05, stable = False, sparse = False, med = False, sv = True):
        """'n_ahead'-step ahead forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast until next `n_ahead` time point.
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
        """
        fit_record = concat_params(self.param_, self.param_names_)
        if type(self.cov_spec_) == LdltConfig:
            forecaster = LdltForecast(
                self.chains_, self.p_, n_ahead, self.y_, sparse, 0.0, fit_record,
                np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                self.fit_intercept, stable, self.thread_, True
            )
        else:
            forecaster = SvForecast(
                self.chains_, self.p_, n_ahead, self.y_, sparse, 0, fit_record,
                np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                self.fit_intercept, stable, self.thread_, sv
            )
        y_distn = forecaster.returnForecast()
        y_distn = process_dens_forecast(y_distn, self.n_features_in_)
        return {
            "forecast": np.median(y_distn, axis=0) if med else np.mean(y_distn, axis=0),
            "se": np.std(y_distn, axis=0, ddof=1),
            "lower": np.quantile(y_distn, level / 2, axis=0),
            "upper": np.quantile(y_distn, 1 - level / 2, axis=0)
        }

    def roll_forecast(self, n_ahead: int, test, level = .05, stable = False, sparse = False, med = False, sv = True):
        """Rolling-window forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast next `n_ahead` time point.
        test : array-like
            Test set to forecast
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
            - "lpl" (float): Average log-predictive likelihood
        """
        fit_record = concat_params(self.param_, self.param_names_)
        test = check_np(test)
        n_horizon = test.shape[0] - n_ahead + 1
        # chunk_size = n_horizon * self.chains_ // self.thread_
        # Check threads and chunk size
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                forecaster = LdltVarRoll(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
            else:
                forecaster = LdltGrpVarRoll(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
        else:
            if self._ggl:
                forecaster = SvVarRoll(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
            else:
                forecaster = SvGrpVarRoll(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
        out_forecast = forecaster.returnForecast()
        # y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), out_forecast.get('forecast')))
        forecast_elem = next(iter(out_forecast.values()))
        y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), forecast_elem))
        return {
            "forecast": np.concatenate(list(map(lambda x: np.median(x, axis = 0), y_distn)), axis = 0) if med else np.concatenate(list(map(lambda x: np.mean(x, axis = 0), y_distn)), axis = 0),
            "se": np.concatenate(list(map(lambda x: np.std(x, axis = 0, ddof=1), y_distn)), axis = 0),
            "lower": np.concatenate(list(map(lambda x: np.quantile(x, level / 2, axis = 0), y_distn)), axis = 0),
            "upper": np.concatenate(list(map(lambda x: np.quantile(x, 1 - level / 2, axis = 0), y_distn)), axis = 0),
            "lpl": out_forecast.get('lpl')
        }

    def expand_forecast(self, n_ahead: int, test, level = .05, stable = False, sparse = False, med = False, sv = True):
        """Expanding-window forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast next `n_ahead` time point.
        test : array-like
            Test set to forecast
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
            - "lpl" (float): Average log-predictive likelihood
        """
        fit_record = concat_params(self.param_, self.param_names_)
        test = check_np(test)
        n_horizon = test.shape[0] - n_ahead + 1
        # chunk_size = n_horizon * self.chains_ // self.thread_
        # Check threads and chunk size
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                forecaster = LdltVarExpand(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
            else:
                forecaster = LdltGrpVarExpand(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
        else:
            if self._ggl:
                forecaster = SvVarExpand(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
            else:
                forecaster = SvGrpVarExpand(
                    self.y_, self.p_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
        out_forecast = forecaster.returnForecast()
        # y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), out_forecast.get('forecast')))
        forecast_elem = next(iter(out_forecast.values()))
        y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), forecast_elem))
        return {
            "forecast": np.concatenate(list(map(lambda x: np.median(x, axis = 0), y_distn)), axis = 0) if med else np.concatenate(list(map(lambda x: np.mean(x, axis = 0), y_distn)), axis = 0),
            "se": np.concatenate(list(map(lambda x: np.std(x, axis = 0, ddof=1), y_distn)), axis = 0),
            "lower": np.concatenate(list(map(lambda x: np.quantile(x, level / 2, axis = 0), y_distn)), axis = 0),
            "upper": np.concatenate(list(map(lambda x: np.quantile(x, 1 - level / 2, axis = 0), y_distn)), axis = 0),
            "lpl": out_forecast.get('lpl')
        }

    def spillover(self):
        pass

    def dynamic_spillover(self):
        pass

class VharBayes(_AutoregBayes):
    """Bayesian Vector Autoregressive Model

    Fits Bayesian VAR model.

    Parameters
    ----------
    data : array-like
        Time series data of which columns indicate the variables
    week : int
        VHAR weekly order, by default 5
    month : int
        VHAR monthly order, by default 22
    n_chain : int
        Number of MCMC chains, by default 1
    n_iter : int
        Number of MCMC total iterations, by default 1000
    n_burn : int
        MCMC burn-in (warm-up), by default `floor(n_iter / 2)`
    n_thin : int
        Thinning every `n_thin`-th iteration, by default 1
    bayes_config : '_BayesConfig'
        Prior configuration, by default SsvsConfig()
    cov_config : {'LdltConfig', 'SvConfig'}
        Prior configuration for covariance matrix, by default LdltConfig()
    intercept_config : 'InterceptConfig'
        Prior configuration for constant term, by default InterceptConfig()
    fit_intercept : bool
        Include constant term in the model, by default True
    minnesota : str
        Minnesota-type group structure
        - "no": Not use the group structure
        - "short": BVAR-minnesota structure
        - "longrun": BVHAR-minnesota structure
    ggl : bool
        If `False`, use group shrinkage parameter instead of global, by default True
    verbose : bool
        If `True`, print progress bar for MCMC, by default False
    n_thread : int
        Number of OpenMP threads, by default 1
    
    Attributes
    ----------
    coef_ : ndarray
        VHAR coefficient matrix.

    intercept_ : ndarray
        VHAR model constant vector.

    n_features_in_ : int
        Number of variables.
    """
    def __init__(
        self,
        data,
        week = 5,
        month = 22,
        n_chain = 1,
        n_iter = 1000,
        n_burn = None,
        n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(),
        fit_intercept = True,
        minnesota = "longrun",
        ggl = True,
        verbose = False,
        n_thread = 1
    ):
        super().__init__(data, month, 3, n_chain, n_iter, n_burn, n_thin, bayes_config, cov_config, intercept_config, fit_intercept, minnesota, ggl)
        self.design_ = build_design(self.y_, week, month, fit_intercept)
        self.response_ = build_response(self.y_, month, month + 1)
        self.week_ = week
        self.month_ = month
        if minnesota == "longrun":
            self._own_id = np.array([2, 4, 6], dtype=np.int32)
            self._cross_id = np.array([1, 3, 5], dtype=np.int32)
        elif minnesota == "short":
            self._own_id = np.array([2], dtype=np.int32)
            self._cross_id = np.array([1, 3, 4], dtype=np.int32)
        else:
            self._own_id = np.array([1], dtype=np.int32)
            self._cross_id = np.array([2], dtype=np.int32)
        self._validate()
        self.thread_ = n_thread
        if self.thread_ > get_maxomp():
            warnings.warn(f"'n_thread' = {self.thread_} is greather than 'omp_get_max_threads()' = {get_maxomp()}. Check with utils.checkomp.get_maxomp(). Check OpenMP support of your machine with utils.checkomp.is_omp().")
        if self.thread_ > n_chain and n_chain != 1:
            warnings.warn(f"'n_thread = {self.thread_} > 'n_chain' = {n_chain}' will not use every thread. Specify as 'n_thread <= 'n_chain'.")
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                self.__model = McmcLdlt(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
            else:
                self.__model = McmcLdltGrp(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
        else:
            if self._ggl:
                self.__model = SvMcmc(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )
            else:
                self.__model = SvGrpMcmc(
                    self.chains_, self.iter_, self.burn_, self.thin_,
                    self.design_, self.response_,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, int(self._prior_type),
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    verbose, self.thread_
                )

    def fit(self):
        """Conduct MCMC and compute posterior mean
        Returns
        -------
        self : object
            An instance of the estimator.
        """
        res = self.__model.returnRecords()
        self.param_names_ = process_record(res, True)
        self.param_ = concat_chain(res, True)
        self.coef_ = self.param_.filter(regex='^phi\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T # -> change name: alpha -> phi
        self.sparse_coef_ = self.param_.filter(regex='^phi_sparse\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T
        if self.fit_intercept:
            self.intercept_ = self.param_.filter(regex='^c\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T
            self.coef_ = np.concatenate([self.coef_, self.intercept_], axis=0)
            self.sparse_coef_ = np.concatenate([self.sparse_coef_, self.param_.filter(regex='^c_sparse\\[[0-9]+\\]').mean().to_numpy().reshape(self.n_features_in_, -1).T], axis=0)
            self.intercept_ = self.intercept_.reshape(self.n_features_in_,)
        self.is_fitted_ = True

    def predict(self, n_ahead: int, level = .05, stable = False, sparse = False, med = False, sv = True):
        """'n_ahead'-step ahead forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast until next `n_ahead` time point.
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
        """
        fit_record = concat_params(self.param_, self.param_names_)
        if type(self.cov_spec_) == LdltConfig:
            forecaster = LdltForecast(
                self.chains_, self.week_, self.month_, n_ahead, self.y_, sparse, 0, fit_record,
                np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                self.fit_intercept, stable, self.thread_, True
            )
        else:
            forecaster = SvForecast(
                self.chains_, self.week_, self.month_, n_ahead, self.y_, sparse, 0, fit_record,
                np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                self.fit_intercept, stable, self.thread_, sv
            )
        y_distn = forecaster.returnForecast()
        y_distn = process_dens_forecast(y_distn, self.n_features_in_)
        return {
            "forecast": np.median(y_distn, axis=0) if med else np.mean(y_distn, axis=0),
            "se": np.std(y_distn, axis=0, ddof=1),
            "lower": np.quantile(y_distn, level / 2, axis=0),
            "upper": np.quantile(y_distn, 1 - level / 2, axis=0)
        }

    def roll_forecast(self, n_ahead: int, test, level = .05, stable = False, sparse = False, med = False, sv = True):
        """Rolling-window forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast next `n_ahead` time point.
        test : array-like
            Test set to forecast
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
            - "lpl" (float): Average log-predictive likelihood
        """
        fit_record = concat_params(self.param_, self.param_names_)
        test = check_np(test)
        n_horizon = test.shape[0] - n_ahead + 1
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                forecaster = LdltVharRoll(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
            else:
                forecaster = LdltGrpVharRoll(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
        else:
            if self._ggl:
                forecaster = SvVharRoll(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
            else:
                forecaster = SvGrpVharRoll(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
        out_forecast = forecaster.returnForecast()
        # y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), out_forecast.get('forecast')))
        forecast_elem = next(iter(out_forecast.values()))
        y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), forecast_elem))
        return {
            "forecast": np.concatenate(list(map(lambda x: np.median(x, axis = 0), y_distn)), axis = 0) if med else np.concatenate(list(map(lambda x: np.mean(x, axis = 0), y_distn)), axis = 0),
            "se": np.concatenate(list(map(lambda x: np.std(x, axis = 0, ddof=1), y_distn)), axis = 0),
            "lower": np.concatenate(list(map(lambda x: np.quantile(x, level / 2, axis = 0), y_distn)), axis = 0),
            "upper": np.concatenate(list(map(lambda x: np.quantile(x, 1 - level / 2, axis = 0), y_distn)), axis = 0),
            "lpl": out_forecast.get('lpl')
        }

    def expand_forecast(self, n_ahead: int, test, level = .05, stable = False, sparse = False, med = False, sv = True):
        """Expanding-window forecasting

        Parameters
        ----------
        n_ahead : int
            Forecast next `n_ahead` time point.
        test : array-like
            Test set to forecast
        level : float
            Level for credible interval, by default .05
        stable : bool
            Filter stable coefficient draws, by default True
        sparse : bool
            Apply restriction to forecasting, by default False
        med : bool
            Use median instead of mean to get point forecast, by default False
        sv : bool
            Use SV term in case of SV model, by default True

        Returns
        -------
        dict
            Density forecasting results
            - "forecast" (ndarray): Posterior mean of forecasting
            - "se" (ndarray): Standard error of forecasting
            - "lower" (ndarray): Lower quantile of forecasting
            - "upper" (ndarray): Upper quantile of forecasting
            - "lpl" (float): Average log-predictive likelihood
        """
        fit_record = concat_params(self.param_, self.param_names_)
        test = check_np(test)
        n_horizon = test.shape[0] - n_ahead + 1
        if type(self.cov_spec_) == LdltConfig:
            if self._ggl:
                forecaster = LdltVharExpand(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
            else:
                forecaster = LdltGrpVharExpand(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, True
                )
        else:
            if self._ggl:
                forecaster = SvVharExpand(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
            else:
                forecaster = SvGrpVharExpand(
                    self.y_, self.week_, self.month_, self.chains_, self.iter_, self.burn_, self.thin_,
                    sparse, 0, fit_record,
                    self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
                    self.init_, self._prior_type,
                    self._group_id, self._own_id, self._cross_id, self.group_,
                    self.fit_intercept, stable, n_ahead, test, True,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_ * n_horizon).reshape(self.chains_, -1).T,
                    np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
                    self.thread_, sv
                )
        out_forecast = forecaster.returnForecast()
        # y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), out_forecast.get('forecast')))
        forecast_elem = next(iter(out_forecast.values()))
        y_distn = list(map(lambda x: process_dens_forecast(x, self.n_features_in_), forecast_elem))
        return {
            "forecast": np.concatenate(list(map(lambda x: np.median(x, axis = 0), y_distn)), axis = 0) if med else np.concatenate(list(map(lambda x: np.mean(x, axis = 0), y_distn)), axis = 0),
            "se": np.concatenate(list(map(lambda x: np.std(x, axis = 0, ddof=1), y_distn)), axis = 0),
            "lower": np.concatenate(list(map(lambda x: np.quantile(x, level / 2, axis = 0), y_distn)), axis = 0),
            "upper": np.concatenate(list(map(lambda x: np.quantile(x, 1 - level / 2, axis = 0), y_distn)), axis = 0),
            "lpl": out_forecast.get('lpl')
        }

    def spillover(self):
        pass

    def dynamic_spillover(self):
        pass