from ..utils._misc import check_np, get_var_intercept
from .._src._ols import OlsVar, OlsVhar

class VarOls:
    def __init__(self, data, lag = 1, fit_intercept = True, method = "nor"):
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
        self.p = lag
        self.fit_intercept = fit_intercept
        self._model = OlsVar(self.y, self.p, self.fit_intercept, self.method)
        # self.fit = {}
        self.coef_ = None
        self.intercept_ = None
        self.cov_ = None
    
    def fit(self):
        fit = self._model.returnOlsRes()
        self.coef_ = fit.get("coefficients")
        self.intercept_ = get_var_intercept(self.coef_, self.p, self.fit_intercept)
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

class VharOls:
    def __init__(self, data, week = 5, month = 22, fit_intercept = True, method = "nor"):
        if method not in ["nor", "chol", "qr"]:
            raise ValueError(f"Argument ('method') '{method}' is not valid: Choose between {['nor', 'chol', 'qr']}")
        self.method = {
            "nor": 1,
            "chol": 2,
            "qr": 3
        }.get(method, None)
        self.y = check_np(data)
        self.n_features_in_ = self.y.shape[1]
        if self.y.shape[0] <= month:
            raise ValueError(f"'data' rows must be larger than `month` = {month}")
        self.week = week
        self.month = month
        self.fit_intercept = fit_intercept
        self._model = OlsVhar(self.y, self.week, self.month, self.fit_intercept, self.method)
        self.coef_ = None
        self.intercept_ = None
        self.cov_ = None
    
    def fit(self):
        fit = self._model.returnOlsRes()
        self.coef_ = fit.get("coefficients")
        self.intercept_ = get_var_intercept(self.coef_, 3, self.fit_intercept)
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