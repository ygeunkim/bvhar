import numpy as np

def check_numeric(data : np.array):
    """Check if the array consists of numeric

    :param data: 2-dim array
    :type data: boolean
    """
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        raise ValueError("All elements should be numeric.")
    # return True

def check_np(data):
    """Check if the dataset is numpy array for Eigen

    :param data: Table-format data
    :type data: Non
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            check_numeric(data)
            return data
        else:
            raise ValueError("Numpy array must be 2-dim.")
    elif isinstance(data, list):
        array_data = np.array(data)
        if array_data.ndim == 2:
            check_numeric(array_data)
            return array_data
        else:
            raise ValueError("np.array(list) should give 2-dim array.")
    else:
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                array_data = data.values
                check_numeric(array_data)
                return array_data
        except ImportError:
            pass
        # Add polars?
        raise ValueError("Unsupported data type.")

def get_var_intercept(coef : np.array, lag: int, fit_intercept : bool):
    dim_design, dim_data = coef.shape
    if not fit_intercept:
        return np.repeat(0, dim_data)
    if dim_design != dim_data * lag + 1:
        ValueError()
    return coef[-1]

def build_grpmat(p, dim_data, minnesota = "longrun"):
    if minnesota not in ["longrun", "short", "no"]:
        raise ValueError(f"Argument ('minnesota')={method} is not valid: Choose between {['longrun', 'short', 'no']}")
    if minnesota == "no":
        return np.full((p * dim_data, dim_data), 1)
    res = [np.identity(dim_data) + 1]
    for i in range(1, p):
        if minnesota == "longrun":
            lag_grp = np.identity(dim_data) + (i * 2 + 1)
        else:
            lag_grp = np.full((dim_data, dim_data), i + 2)
        res.append(lag_grp)
    return np.vstack(res)