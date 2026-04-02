# Overview

`bvhar` is header-only `C++` library for multivariate time series analysis.
It is used in both `R` and `Python`.

- [bvhar for R](https://bvhar.baeconverse.org)
- [bvhar for Python](https://bvhar.baeconverse.org/python/)

## For C++ developers

(In preparation)

## For R developers

`R` developers can use the headers through [`Rcpp`](https://www.rcpp.org).
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

You need to add `plugins` attribute because the header in this package should define `BVHAR_USE_RCPP` macro.
If the `BVHAR_USE_RCPP` macro is not defined, the headers does not import `Rcpp`.

## For Python developers

(In preparation)
