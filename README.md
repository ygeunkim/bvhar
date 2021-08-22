
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bvhar <img src='man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![Codecov test
coverage](https://codecov.io/gh/ygeunkim/bvhar/branch/master/graph/badge.svg)](https://codecov.io/gh/ygeunkim/bvhar?branch=master)
<!-- badges: end -->

## Overview

`bvhar` provides functions to analyze multivariate time series time
series using

-   VAR
-   VHAR (Vector HAR)
-   BVAR (Bayesian VAR)
-   **BVHAR (Bayesian VHAR)**

Basically, the package focuses on the research with forecasting.

## Installation

You can only install the development version at this point.

``` r
# install.packages("remotes")
remotes::install_github("ygeunkim/bvhar")
```

## Usage

``` r
library(bvhar)
library(dplyr)
```

### VAR

Out-of-sample forecasting:

``` r
h <- 19
etf_tr <-
  etf_vix %>%
  slice(seq_len(n() - h))
#-------------------
etf_te <- setdiff(etf_vix, etf_tr)
```

VAR(5):

``` r
mod_var <- var_lm(etf_tr, 5)
```

Forecasting:

``` r
forecast_var <- predict(mod_var, h)$forecast
```

MSE:

``` r
(msevar <- apply(etf_te - forecast_var, 2, function(x) mean(x^2)))
#>    EVZCLS    GVZCLS    OVXCLS  VXEEMCLS  VXEWZCLS  VXFXICLS  VXGDXCLS  VXSLVCLS 
#>  1.625078  4.009469 71.268936  3.012375  1.911313  3.818629  4.603957 10.266017 
#>  VXXLECLS 
#> 38.117087
```

### VHAR

``` r
mod_vhar <- vhar_lm(etf_tr)
```

MSE:

``` r
forecast_vhar <- predict(mod_vhar, h)$forecast
(msevhar <- apply(etf_te - forecast_vhar, 2, function(x) mean(x^2)))
#>    EVZCLS    GVZCLS    OVXCLS  VXEEMCLS  VXEWZCLS  VXFXICLS  VXGDXCLS  VXSLVCLS 
#>  2.574746  6.387274 70.829157  4.321342  3.086089  5.341502  4.378567  7.636454 
#>  VXXLECLS 
#> 52.135727
```

### BVAR

Minnesota prior:

``` r
lam <- .2
delta <- rep(0, ncol(etf_vix)) # litterman
sig <- apply(etf_tr, 2, sd)
eps <- 1e-04
```

``` r
mod_bvar <- bvar_minnesota(etf_tr, 5, sig, lam, delta, eps)
```

MSE:

``` r
forecast_bvar <- predict(mod_bvar, h)$forecast
(msebvar <- apply(etf_te - forecast_bvar, 2, function(x) mean(x^2)))
#>    EVZCLS    GVZCLS    OVXCLS  VXEEMCLS  VXEWZCLS  VXFXICLS  VXGDXCLS  VXSLVCLS 
#>  1.675226  3.007671 53.085572  4.233823  5.998921  3.847304  4.247299  8.387925 
#>  VXXLECLS 
#> 42.483580
```

### BVHAR

``` r
mod_bvhar <- bvhar_minnesota(etf_tr, sigma = sig, lambda = lam, delta = delta, eps = eps)
```

MSE:

``` r
forecast_bvhar <- predict(mod_bvhar, h)$forecast
(msebvhar <- apply(etf_te - forecast_bvhar, 2, function(x) mean(x^2)))
#>    EVZCLS    GVZCLS    OVXCLS  VXEEMCLS  VXEWZCLS  VXFXICLS  VXGDXCLS  VXSLVCLS 
#>  1.707028  3.453921 48.813705  3.388200  8.974295  5.351510  4.725564  6.815297 
#>  VXXLECLS 
#> 41.515688
```

Comparing:

``` r
# VAR---------------
mean(msevar)
#> [1] 15.40365
# VHAR--------------
mean(msevhar)
#> [1] 17.4101
# BVAR--------------
mean(msebvar)
#> [1] 14.10748
# BVHAR-------------
mean(msebvhar)
#> [1] 13.86058
```
