from ..utils._misc import check_np
from ._ols import OlsVar, OlsVhar

class VarOls:
    def __init__(self, data, lag = 1, include_mean = True, method = "nor"):
        if method not in ["nor", "chol", "qr"]:
            raise ValueError(f"Argument ('method') '{method}' is not valid: Choose between {['nor', 'chol', 'qr']}")
        self.method = {
            "nor": 1,
            "chol": 2,
            "qr": 3
        }.get(method, None)
        self.y = check_np(data)
        self.p = lag
        self.constant = include_mean
        self._model = OlsVar(self.y, self.p, self.constant, self.method)
        # self.fit = {}
        self.coef_ = None
    
    def fit(self):
        fit = self._model.return_ols_res()
        self.coef_ = fit.get("coefficients")

class VharOls:
    def __init__(self, data, week = 5, month = 22, include_mean = True, method = "nor"):
        if method not in ["nor", "chol", "qr"]:
            raise ValueError(f"Argument ('method') '{method}' is not valid: Choose between {['nor', 'chol', 'qr']}")
        self.method = {
            "nor": 1,
            "chol": 2,
            "qr": 3
        }.get(method, None)
        self.y = check_np(data)
        self.week = week
        self.month = month
        self.constant = include_mean
        self._model = OlsVhar(self.y, self.week, self.month, self.constant, self.method)
        self.coef_ = None
    
    def fit(self):
        fit = self._model.return_ols_res()
        self.coef_ = fit.get("coefficients")

__all__ = [
    "VarOls",
    "VharOls"
]