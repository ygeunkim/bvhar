# Overview

This package provides public C++ header.
Package developers or `Rcpp` can use these easily.
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
