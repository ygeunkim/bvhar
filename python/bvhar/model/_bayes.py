from ..utils._misc import make_fortran_array, check_np, build_grpmat, process_record, concat_chain
from ..utils.checkomp import get_maxomp
from .._src._design import build_response, build_design
from .._src._ldlt import McmcLdlt
from ._spec import LdltConfig, SvConfig, InterceptConfig
from ._spec import BayesConfig, SsvsConfig, HorseshoeConfig, MinnesotaConfig, DlConfig, NgConfig
import numpy as np
import pandas as pd
import warnings
from math import floor

class AutoregBayes:
    def __init__(
        self, data, lag, p, n_chain = 1, n_iter = 1000,
        n_burn = lambda n_iter: floor(n_iter / 2),
        n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(), fit_intercept = True,
        minnesota = "longrun"
    ):
        self.y = check_np(data)
        self.n_features_in_ = self.y.shape[1]
        self.p_ = p # 3 in VHAR
        self.lag_ = lag # month in VHAR
        if self.y.shape[0] <= self.lag_:
            raise ValueError(f"'data' rows must be larger than 'lag' = {self.lag_}")
        # self.design_ = build_design(self.y, self.lag_, fit_intercept)
        # self.response_ = build_response(self.y, self.lag_, self.lag_ + 1)
        self.chains_ = int(n_chain)
        self.iter_ = int(n_iter)
        self.burn_ = int(n_burn)
        self.thin_ = int(n_thin)
        self.fit_intercept = fit_intercept
        self.group_ = build_grpmat(self.p_, self.n_features_in_, minnesota)
        self._group_id = pd.unique(self.group_.flatten(order='F')).astype(np.int32)
        self._own_id = None
        self._cross_id = None
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
            pass
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
                    'contem_slab': init_contem_slab
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
        self.init_ = make_fortran_array(self.init_)
        self._prior_type = {
            "Minnesota": 1,
            "SSVS": 2,
            "Horseshoe": 3,
            "HMN": 4,
            "NG": 5,
            "DL": 6
        }.get(self.spec_.prior)
        self.coef_ = None
        self.intercept_ = None
        self.param_names_ = None
        self.param_ = None
        # self.cov_ = None
    
    def _validate(self):
        if not isinstance(self.cov_spec_, LdltConfig):
            raise TypeError("`cov_config` should be `LdltConfig` or `SvConfig`.")
        if not isinstance(self.intercept_spec_, InterceptConfig):
            raise TypeError("`intercept_config` should be `InterceptConfig` when 'fit_intercept' is True.")
        if not isinstance(self.spec_, BayesConfig):
            raise TypeError("`bayes_spec` should be the derived class of `BayesConfig`.")
        self.cov_spec_.update(self.n_features_in_)
        self.intercept_spec_.update(self.n_features_in_)
        if type(self.spec_) == SsvsConfig:
            self.spec_.update(self._group_id, self._own_id, self._cross_id)
        elif type(self.spec_) == HorseshoeConfig:
            pass
        elif type(self.spec_) == MinnesotaConfig:
            self.spec_.update(self.y, self.p_, self.n_features_in_)
        elif type(self.spec_) == DlConfig:
            pass
        elif type(self.spec_) == NgConfig:
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

class VarBayes(AutoregBayes):
    def __init__(
        self,
        data,
        lag = 1,
        n_chain = 1,
        n_iter = 1000,
        n_burn = lambda n_iter: floor(n_iter / 2),
        n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(),
        fit_intercept = True,
        minnesota = True,
        verbose = False,
        n_thread = 1
    ):
        super().__init__(data, lag, lag, n_chain, n_iter, n_burn, n_thin, bayes_config, cov_config, intercept_config, fit_intercept, "short" if minnesota else "no")
        design = build_design(self.y, lag, fit_intercept)
        response = build_response(self.y, lag, lag + 1)
        if minnesota:
            self._own_id = np.array([2], dtype=np.int32)
            self._cross_id = np.arange(1, self.p_ + 2, dtype=np.int32)
            self._cross_id = np.delete(self._cross_id, 1)
        else:
            self._own_id = np.array([2], dtype=np.int32)
            self._cross_id = np.array([2], dtype=np.int32)
        self._validate()
        if n_thread > get_maxomp():
            warnings.warn(f"'n_thread' = {n_thread} is greather than 'omp_get_max_threads()' = {get_maxomp()}. Check with utils.checkomp.get_maxomp(). Check OpenMP support of your machine with utils.checkomp.is_omp().")
        if n_thread > n_chain and n_chain != 1:
            warnings.warn(f"'n_thread = {n_thread} > 'n_chain' = {n_chain}' will not use every thread. Specify as 'n_thread <= 'n_chain'.")
        self.__model = McmcLdlt(
            self.chains_, self.iter_, self.burn_, self.thin_,
            design, response,
            self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
            self.init_, int(self._prior_type),
            self._group_id, self._own_id, self._cross_id, self.group_,
            self.fit_intercept,
            np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
            verbose, n_thread
        )

    def fit(self):
        res = self.__model.returnRecords()
        self.param_names_ = process_record(res)
        self.param_ = concat_chain(res)
        self.coef_ = self.param_.filter(regex='^alpha\\[[0-9]+\\]').mean().to_numpy().reshape(3, -1).T
        if self.fit_intercept:
            self.intercept_ = self.param_.filter(regex='^c\\[[0-9]+\\]').mean().to_numpy().reshape(3, -1).T
            self.coef_ = np.concatenate([self.coef_, self.intercept_], axis=0)
            self.intercept_ = self.intercept_.reshape(self.n_features_in_,)

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

class VharBayes(AutoregBayes):
    def __init__(
        self,
        data,
        week = 5,
        month = 22,
        n_chain = 1,
        n_iter = 1000,
        n_burn = lambda n_iter: floor(n_iter / 2),
        n_thin = 1,
        bayes_config = SsvsConfig(),
        cov_config = LdltConfig(),
        intercept_config = InterceptConfig(),
        fit_intercept = True,
        minnesota = "longrun",
        verbose = False,
        n_thread = 1
    ):
        super().__init__(data, month, 3, n_chain, n_iter, n_burn, n_thin, bayes_config, cov_config, intercept_config, fit_intercept, minnesota)
        design = build_design(self.y, week, month, fit_intercept)
        response = build_response(self.y, month, month + 1)
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
        if n_thread > get_maxomp():
            warnings.warn(f"'n_thread' = {n_thread} is greather than 'omp_get_max_threads()' = {get_maxomp()}. Check with utils.checkomp.get_maxomp(). Check OpenMP support of your machine with utils.checkomp.is_omp().")
        if n_thread > n_chain and n_chain != 1:
            warnings.warn(f"'n_thread = {n_thread} > 'n_chain' = {n_chain}' will not use every thread. Specify as 'n_thread <= 'n_chain'.")
        self.__model = McmcLdlt(
            self.chains_, self.iter_, self.burn_, self.thin_,
            design, response,
            self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
            self.init_, int(self._prior_type),
            self._group_id, self._own_id, self._cross_id, self.group_,
            self.fit_intercept,
            np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
            verbose, n_thread
        )

    def fit(self):
        res = self.__model.returnRecords()
        self.param_names_ = process_record(res)
        self.param_ = concat_chain(res)
        self.coef_ = self.param_.filter(regex='^alpha\\[[0-9]+\\]').mean().to_numpy().reshape(3, -1).T # -> change name: alpha -> phi
        if self.fit_intercept:
            self.intercept_ = self.param_.filter(regex='^c\\[[0-9]+\\]').mean().to_numpy().reshape(3, -1).T
            self.coef_ = np.concatenate([self.coef_, self.intercept_], axis=0)
            self.intercept_ = self.intercept_.reshape(self.n_features_in_,)

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