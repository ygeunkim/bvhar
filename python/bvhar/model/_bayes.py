from ..utils._misc import check_np, get_var_intercept, build_grpmat
from ..src._design import build_response, build_design
from .._src._ldlt import McmcLdlt
from ._spec import LdltConfig, SvConfig, InterceptConfig
from ._spec import BayesConfig, SsvsConfig, HorseshoeConfig, MinnesotaConfig, DlConfig, NgConfig
import numpy as np
import pandas as pd
from math import floor

class VarBayes:
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
        self.y = check_np(data)
        self.n_features_in_ = self.y.shape[1]
        # if self.y.shape[0] <= lag:
        #     raise ValueError(f"'data' rows must be larger than `lag` = {lag}")
        design = build_design(self.y, lag, fit_intercept)
        response = build_response(self.y, lag, lag + 1)
        self.p_ = lag
        self.chains_ = n_chain
        self.iter_ = n_iter
        self.burn_ = n_burn
        self.thin_ = n_thin
        self.fit_intercept = fit_intercept
        minnesota = "short" if minnesota else "no"
        self.group_ = build_grpmat(self.p_, self.n_features_in_, minnesota)
        # self.__group_id = np.unique(self.group.flatten(order='F'))
        self.__group_id = pd.unique(self.group_.flatten(order='F'))
        self.__own_id = None
        self.__cross_id = None
        n_grp = len(self.__group_id)
        n_alpha = self.n_features_in_ * self.n_features_in_ * self.p_
        n_design = design.shape[1]
        n_eta = self.n_features_in_ * (self.n_features_in_ - 1) / 2
        if minnesota:
            self.__own_id = np.array([2])
            self.__cross_id = np.arange(1, self.p_ + 2)
            self.__cross_id = np.delete(self.__cross_id, 1)
        else:
            self.__own_id = np.array([2])
            self.__cross_id = np.array([2])
        self.cov_spec_ = cov_config
        self.spec_ = bayes_config
        self.intercept_spec_ = intercept_config
        self.validate()
        self.init_ = [
            {
                'init_coef': np.random.uniform(-1, 1, (self.n_features_in_, n_design)),
                'init_contem': np.exp(np.random.uniform(-1, 0, n_eta))
            }
            for _ in range(num_chains)
        ]
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
                    'contem_global_sparsity': contem_global_sparsity
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
                    'contem_global_sparsity': contem_global_sparsity
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
                    'contem_global_sparsity': contem_global_sparsity
                })
        self.__model = McmcLdlt(
            self.chains_, self.iter_, self.burn_, self.thin_,
            design, response,
            self.cov_spec_.to_dict(), self.spec_.to_dict(), self.intercept_spec_.to_dict(),
            self.init_, 1,
            self.__group_id, self.__own_id, self.__cross_id, self.group_,
            self.fit_intercept,
            np.random.randint(low = 1, high = np.iinfo(np.int32).max, size = self.chains_),
            verbose, n_thread
        )
        self.coef_ = None
        self.intercept_ = None
        # self.cov_ = None
    
    def validate(self):
        if self.y.shape[0] <= self.p_:
            raise ValueError(f"'data' rows must be larger than `lag` = {self.p_}")
        if not isinstance(self.cov_spec_, LdltConfig):
            raise TypeError("`cov_config` should be `LdltConfig` or `SvConfig`.")
        if not isinstance(self.intercept_spec_, InterceptConfig):
            raise TypeError("`intercept_config` should be `InterceptConfig`.")
        if not isinstance(self.spec_, BayesConfig):
            raise TypeError("`bayes_spec` should be the derived class of `BayesConfig`.")
        self.cov_spec_.update(self.n_features_in_)
        self.intercept_spec_.update(self.n_features_in_)
        if type(self.spec_) == SsvsConfig:
            self.spec_.update(self.__group_id, self.__own_id, self.__cross_id)
        elif type(self.spec_) == HorseshoeConfig:
            pass
        elif type(self.spec_) == MinnesotaConfig:
            pass
        elif type(self.spec_) == DlConfig:
            pass
        elif type(self.spec_) == NgConfig:
            pass

    def fit(self):
        fit = self.__model.returnRecords()
        self.coef_ = fit[0].get("d_record") # temporary

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
