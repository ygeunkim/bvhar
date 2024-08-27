# bvhar <img src='../man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->
<!-- badges: end -->

Started to develop `bvhar` in Python!

## Installation

In `python` directory:

``` bash
pip install -e .
```

Check OpenMP:

``` python
from bvhar.utils import checkomp
checkomp.check_omp()
```

    OpenMP threads:  16
