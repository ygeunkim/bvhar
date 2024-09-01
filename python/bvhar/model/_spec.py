import numpy as np

class LdltConfig:
    def __init__(self, ig_shape = 3, ig_scale = .01):
        self.process = "Homoskedastic"
        self.prior = "Cholesky"
        self.shape = self.validate(ig_shape, "ig_shape")
        self.scale = self.validate(ig_scale, "ig_scale")
    
    def validate(self, value, member):
        if isinstance(value, int):
            return [float(value)]
        elif isinstance(value, (float, np.number)):
            return [value]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            return np.array(value)
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, n_dim: int):
        if len(self.shape) == 1:
            self.shape = np.repeat(self.shape, n_dim)
        if len(self.scale) == 1:
            self.scale = np.repeat(self.scale, n_dim)

    def to_dict(self):
        return {
            # "process": self.process,
            # "prior": self.prior,
            "shape": self.shape,
            "scale": self.scale
        }

class SvConfig(LdltConfig):
    def __init__(self, ig_shape = 3, ig_scale = .01, initial_mean = 1, initial_prec = .1):
        super().__init__(ig_shape, ig_scale)
        self.process = "SV"
        self.initial_mean = self.validate(initial_mean, "initial_mean", 1)
        self.initial_prec = self.validate(initial_prec, "initial_prec", 2)
    
    def validate(self, value, member, n_dim):
        if isinstance(value, (int, float, np.number)):
            return [value] if n_dim == 1 else value * np.identity(1)
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member} cannot be empty.")
            value_array = np.array(value)
            if value_array.ndim > 2:
                raise ValueError(f"'{member} has wrong dim = {n_dim}.")
            elif value_array.ndim != n_dim:
                raise ValueError(f"'{member}' should be {n_dim}-dim.")
            return value_array
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, n_dim: int):
        super().update(n_dim)
        if isinstance(self.initial_mean, (int, float, np.number)) or (hasattr(self.initial_mean, '__len__') and len(self.initial_mean) == 1):
            self.initial_mean = np.repeat(self.initial_mean, n_dim)
        if isinstance(self.initial_prec, (int, float, np.number)) or (hasattr(self.initial_prec, '__len__') and len(self.initial_prec) == 1):
            self.initial_prec = self.initial_prec[0] * np.identity(n_dim)

    def to_dict(self):
        res = super().to_dict()
        # res["process"] = self.process
        res["initial_mean"] = self.initial_mean
        res["initial_prec"] = self.initial_prec
        return res

class InterceptConfig:
    def __init__(self, mean = 0, sd = .1):
        self.process = "Intercept"
        self.prior = "Normal"
        self.mean_non = self.validate(mean, "mean")
        self.sd_non = self.validate(sd, "sd")
    
    def validate(self, value, member):
        # if isinstance(value, int):
        #     return [float(value)]
        # elif isinstance(value, (float, np.number)):
        #     return [value]
        if isinstance(value, int):
            return float(value)
        if isinstance(value, (float, np.number)):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            return np.array(value)
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, n_dim: int):
        if isinstance(self.mean_non, (int, float, np.number)) or (hasattr(self.mean_non, '__len__') and len(self.mean_non) == 1):
            self.mean_non = np.repeat(self.mean_non, n_dim)
        if not isinstance(self.sd_non, (int, float, np.number)) or (hasattr(self.sd_non, '__len__') and len(self.sd_non) > 1):
            raise ValueError("'sd_non' should be a number.")

    def to_dict(self):
        return {
            # "process": self.process,
            # "prior": self.prior,
            "mean_non": self.mean_non,
            "sd_non": self.sd_non
        }

class BayesConfig:
    def __init__(self, prior):
        self.prior = prior
    
    def validate(self):
        pass

    def update(self):
        pass

    def to_dict(self):
        # return {"prior": self.prior}
        pass

class SsvsConfig(BayesConfig):
    def __init__(
        self, coef_spike_scl = .01, coef_slab_shape = .01, coef_slab_scl = .01, coef_s1 = [1, 1], coef_s2 = [1, 1],
        chol_spike_scl = .01, chol_slab_shape = .01, chol_slab_scl = .01,
        chol_s1 = 1, chol_s2 = 1
    ):
        super().__init__("SSVS")
        self.coef_spike_scl = self.validate(coef_spike_scl, "coef_spike_scl")
        self.coef_slab_shape = self.validate(coef_slab_shape, "coef_slab_shape")
        self.coef_slab_scl = self.validate(coef_slab_scl, "coef_slab_scl")
        self.coef_s1 = self.validate(coef_s1, "coef_s1", 2)
        self.coef_s2 = self.validate(coef_s2, "coef_s1", 2)
        self.chol_spike_scl = self.validate(chol_spike_scl, "chol_spike_scl")
        self.chol_slab_shape = self.validate(chol_slab_shape, "chol_slab_shape")
        self.chol_slab_scl = self.validate(chol_slab_scl, "chol_slab_scl")
        self.chol_s1 = self.validate(chol_s1, "chol_s1")
        self.chol_s2 = self.validate(chol_s2, "chol_s1")

    def validate(self, value, member, n_size = None):
        # if isinstance(value, int):
        #     return [float(value)]
        # elif isinstance(value, (float, np.number)):
        #     return [value]
        if isinstance(value, (int, float, np.number)):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            elif n_size is not None and len(value) != n_size:
                raise ValueError(f"'{member}' length must be {n_size}.")
            value_array = np.array(value)
            if value_array.ndim > 2:
                raise ValueError(f"'{member} has wrong dim = {value_array.ndim}.")
            return value_array
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")

    def update(self, grp_id: np.array, own_id: np.array, cross_id: np.array):
        if len(self.coef_s1) == 2:
            coef_s1 = np.zeros(len(grp_id))
            coef_s1[np.isin(grp_id, own_id)] = self.coef_s1[0]
            coef_s1[np.isin(grp_id, cross_id)] = self.coef_s1[1]
            self.coef_s1 = coef_s1
        if len(self.coef_s2) == 2:
            coef_s2 = np.zeros(len(grp_id))
            coef_s2[np.isin(grp_id, own_id)] = self.coef_s2[0]
            coef_s2[np.isin(grp_id, cross_id)] = self.coef_s2[1]
            self.coef_s2 = coef_s2

    def to_dict(self):
        return {
            "coef_spike_scl": self.coef_spike_scl,
            "coef_slab_shape": self.coef_slab_shape,
            "coef_slab_scl": self.coef_slab_scl,
            "coef_s1": self.coef_s1,
            "coef_s2": self.coef_s2,
            "chol_spike_scl": self.chol_spike_scl,
            "chol_slab_shape": self.chol_slab_shape,
            "chol_slab_scl": self.chol_slab_scl,
            "chol_s1": self.chol_s1,
            "chol_s2": self.chol_s2
        }

class HorseshoeConfig(BayesConfig):
    def __init__(self):
        super().__init__("Horseshoe")
    
    def validate(self):
        pass

    def update(self):
        pass

    def to_dict(self):
        # return {"prior": self.prior}
        return dict()

class MinnesotaConfig(BayesConfig):
    def __init__(self):
        super().__init__("Minnesota")
    
    def validate(self):
        pass

    def update(self):
        pass

    def to_dict(self):
        # return {"prior": self.prior}
        pass

class DlConfig(BayesConfig):
    def __init__(self):
        super().__init__("DL")
    
    def validate(self):
        pass

    def update(self):
        pass

    def to_dict(self):
        # return {"prior": self.prior}
        pass

class NgConfig(BayesConfig):
    def __init__(self):
        super().__init__("NG")
    
    def validate(self):
        pass

    def update(self):
        pass

    def to_dict(self):
        # return {"prior": self.prior}
        pass