import os
from .utils.checkomp import is_omp

if is_omp():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK']='True'