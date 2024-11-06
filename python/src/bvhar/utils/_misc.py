import re
import numpy as np
import pandas as pd

def check_numeric(data : np.array):
    """Check if the array consists of numeric

    Parameters
    ----------
    data : np.array
        2-dim array

    Raises
    ------
    ValueError
        If the array does not consists of only numeric values.
    """
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        raise ValueError("All elements should be numeric.")

# """Check if the array consists of numeric

#     :param data: 2-dim array
#     :type data: boolean
#     """

def make_fortran_array(object):
    """Make the array for Eigen input

    Parameters
    ----------
    object : np.array
        Array to be used as a input for Eigen::Matrix

    Returns
    -------
    array
        Array available for Eigen::Matrix input
    """
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

    Parameters
    ----------
    data : None
        Table-format data

    Returns
    -------
    array
        Result of :func:`make_fortran_array`

    Raises
    ------
    ValueError
        If the array.ndim is not 2.
    ValueError
        If the list.ndim is not 2.
    ValueError
        If the data is not array, list, nor pd.DataFrame.
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
        if isinstance(data, pd.DataFrame):
            array_data = data.values
            check_numeric(array_data)
            return make_fortran_array(array_data)
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

def process_record(record_list: list, har = False):
    rec_names = list(record_list[0].keys())
    if har:
        return [re.sub(r'alpha_', 'phi_', name).replace('_record', '') for name in rec_names]
    return [name.replace('_record', '') for name in rec_names]

def concat_chain(record_list: list, har = False):
    record_concat = pd.DataFrame()
    tot_draw_n = 0
    for chain_id, chain_dict in enumerate(record_list):
        param_record = pd.DataFrame()
        for rec_names, record in chain_dict.items():
            if har:
                param = re.sub(r'alpha_', 'phi_', rec_names).replace('_record', '')
            else:
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

def concat_params(record: pd.DataFrame, param_names: str):
    res = {}
    # n_chains = record['_chain'].nunique()
    for _name in param_names:
        param_columns = [col for col in record.columns if col.startswith(_name)]
        param_record = record[param_columns + ['_chain']]
        param_record_chain = [df for _, df in param_record.groupby('_chain')]
        array_chain = [df.drop('_chain', axis=1).values for df in param_record_chain]
        res[f"{_name}_record"] = array_chain
    return res

def process_dens_forecast(pred_list: list, n_dim: int):
    # shape_pred = pred_list[0].shape # (step, dim * draw)
    # n_ahead = shape_pred[0]
    # n_draw = int(shape_pred[1] / n_dim)
    n_draw = int(pred_list[0].shape[1] / n_dim)
    res = []
    for arr in pred_list:
        res.append([arr[:, range(id * n_dim, id * n_dim + n_dim)] for id in range(n_draw)])
    return np.concatenate(res, axis=0)
