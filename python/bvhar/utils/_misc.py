import numpy as np
import pandas as pd

def check_numeric(data : np.array):
    """Check if the array consists of numeric

    :param data: 2-dim array
    :type data: boolean
    """
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        raise ValueError("All elements should be numeric.")
    # return True

# def __make_fortran_array(arr: np.array):
#     if not arr.flags.f_contiguous or arr.dtype != np.float64:
#         return np.asfortranarray(arr, dtype=np.float64)
#     return arr

def make_fortran_array(object):
    # if not arr.flags.f_contiguous or arr.dtype != np.float64:
    #     return np.asfortranarray(arr, dtype=np.float64)
    # return arr
    if isinstance(object, np.ndarray):
        if not object.flags.f_contiguous or object.dtype != np.float64:
            return np.asfortranarray(object, dtype=np.float64)
    elif isinstance(object, dict):
        return {k: make_fortran_array(v) for k, v in object.items()}
    elif isinstance(object, list):
        return [make_fortran_array(item) for item in object]
    return object

def check_np(data):
    """Check if the dataset is numpy array for Eigen

    :param data: Table-format data
    :type data: Non
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            check_numeric(data)
            return make_fortran_array(data)
        else:
            raise ValueError("Numpy array must be 2-dim.")
    elif isinstance(data, list):
        array_data = np.array(data)
        if array_data.ndim == 2:
            check_numeric(array_data)
            return make_fortran_array(array_data)
        else:
            raise ValueError("np.array(list) should give 2-dim array.")
    else:
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                array_data = data.values
                check_numeric(array_data)
                return make_fortran_array(array_data)
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

def process_record(record_list: list):
    rec_names = list(record_list[0].keys())
    return [name.replace('_record', '') for name in rec_names]

def concat_chain(record_list: list):
    record_concat = pd.DataFrame()
    tot_draw_n = 0
    for chain_id, chain_dict in enumerate(record_list):
        param_record = pd.DataFrame()
        for rec_names, record in chain_dict.items():
            param = rec_names.replace('_record', '')
            n_col = record.shape[1]
            chain_record = pd.DataFrame(
                record,
                columns=[param + f"[{i}]" for i in range(1, n_col + 1)]
            )
            n_draw = len(chain_record)
            param_record = pd.concat([param_record, chain_record], axis=1)
        param_record['_chain'] = chain_id
        param_record['_iteration'] = range(1, n_draw + 1)
        param_record['_draw'] = range(tot_draw_n + 1, tot_draw_n + n_draw + 1)
        tot_draw_n += n_draw
        record_concat = pd.concat([record_concat, param_record], axis=0)
    return record_concat