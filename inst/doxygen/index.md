# Overview

This package provides public `C++` header.
It is used in both `R` and `Python`.

- [bvhar for R](https://ygeunkim.github.io/package/bvhar/)
- [bvhar for Python](https://ygeunkim.github.io/package/bvhar/python/)

## For R package developers

`R` package developers can use the headers through [`Rcpp`](https://www.rcpp.org).
You can use these by writing in your R package `DESCRIPTION`.

```
LinkingTo: 
    BH,
    Rcpp,
    RcppEigen,
    RcppSpdlog,
    RcppThread,
    bvhar
```

Also, you can use in your single `C++` source:

```cpp
// [[Rcpp::depends(BH, RcppEigen, RcppSpdlog, RcppThread, bvhar)]]
// [[Rcpp::plugins(bvhar)]]

// [[Rcpp::export]]
// Your C++ code
```

You need to add `plugins` attribute because the header in this package should define `USE_RCPP` macro.
Or you can use instead:

```r
Sys.setenv("PKG_CPPFLAGS" = "-DUSE_RCPP")
```

If the `USE_RCPP` macro is not defined, the headers are compiled for [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html) of `Python`.
