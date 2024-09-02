from ..utils._misc import check_np, get_var_intercept
from .._src._ols import OlsVar, OlsVhar

class Vectorautoreg:
    def __init__(self, data, lag, p, fit_intercept = True, method = "nor"):
        if method not in ["nor", "chol", "qr"]:
            raise ValueError(f"Argument ('method') '{method}' is not valid: Choose between {['nor', 'chol', 'qr']}")
        self.method = {
            "nor": 1,
            "chol": 2,
            "qr": 3
        }.get(method, None)
        self.y = check_np(data)
        self.n_features_in_ = self.y.shape[1]
        if self.y.shape[0] <= lag:
            raise ValueError(f"'data' rows must be larger than `lag` = {lag}")
        # self.p = lag
        self.p_ = p # 3 in VHAR
        self.lag_ = lag # month in VHAR
        if self.y.shape[0] <= self.lag_:
            raise ValueError(f"'data' rows must be larger than 'lag' = {self.lag_}")
        self.fit_intercept = fit_intercept
        self._model = None
        self.coef_ = None
        self.intercept_ = None
        self.cov_ = None

    def fit(self):
        fit = self._model.returnOlsRes()
        self.coef_ = fit.get("coefficients")
        self.intercept_ = get_var_intercept(self.coef_, self.p_, self.fit_intercept)
        self.cov_ = fit.get("covmat")

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

class VarOls(Vectorautoreg):
    def __init__(self, data, lag = 1, fit_intercept = True, method = "nor"):
        super().__init__(data, lag, lag, fit_intercept, method)
        self._model = OlsVar(self.y, self.p_, self.fit_intercept, self.method)
    
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

class VharOls(Vectorautoreg):
    def __init__(self, data, week = 5, month = 22, fit_intercept = True, method = "nor"):
        super().__init__(data, month, 3, fit_intercept, method)
        self.week_ = week
        self.month_ = self.lag_ # or self.lag_ = [week, month]
        self._model = OlsVhar(self.y, week, self.lag_, self.fit_intercept, self.method)
    
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