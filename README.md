
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bvhar <img src='man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R build
status](https://github.com/ygeunkim/bvhar/workflows/R-CMD-check/badge.svg)](https://github.com/ygeunkim/bvhar/actions?workflow=R-CMD-check)
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

## Models

``` r
library(bvhar) # this package
library(dplyr)
```

Repeatedly, `bvhar` is a research tool to analyze multivariate time
series model above

| Model |     function      |     S3     |
|:-----:|:-----------------:|:----------:|
|  VAR  |     `var_lm`      |  `varlse`  |
| VHAR  |     `vhar_lm`     | `vharlse`  |
| BVAR  | `bvar_minnesota`  |  `bvarmn`  |
| BVAR  |    `bvar_flat`    | `bvarflat` |
| BVHAR | `bvhar_minnesota` | `bvharmn`  |

As the other analyzer tools use S3 such as `lm`, this package use
methods `coef`, `predict`, etc. This readme document shows out-of-sample
forecasting briefly. Details about each function are in vignettes and
help documents.

Out-of-sample forecasting:

``` r
h <- 19
etf_split <- divide_ts(etf_vix, h) # Try ?divide_ts
etf_tr <- etf_split$train
etf_te <- etf_split$test
```

### VAR

VAR(5):

``` r
mod_var <- var_lm(etf_tr, 5)
```

Forecasting:

``` r
forecast_var <- predict(mod_var, h)
```

MSE:

``` r
(msevar <- mse(forecast_var, etf_te))
#>   EVZCLS   GVZCLS   OVXCLS VXEEMCLS VXEWZCLS VXFXICLS VXGDXCLS VXSLVCLS 
#>     1.63     4.01    71.27     3.01     1.91     3.82     4.60    10.27 
#> VXXLECLS 
#>    38.12
```

### VHAR

``` r
mod_vhar <- vhar_lm(etf_tr)
```

MSE:

``` r
forecast_vhar <- predict(mod_vhar, h)
(msevhar <- mse(forecast_vhar, etf_te))
#>   EVZCLS   GVZCLS   OVXCLS VXEEMCLS VXEWZCLS VXFXICLS VXGDXCLS VXSLVCLS 
#>     2.57     6.39    70.83     4.32     3.09     5.34     4.38     7.64 
#> VXXLECLS 
#>    52.14
```

### BVAR

Minnesota prior:

``` r
lam <- .3
delta <- rep(1, ncol(etf_vix)) # litterman
sig <- apply(etf_tr, 2, sd)
eps <- 1e-04
```

``` r
mod_bvar <- bvar_minnesota(etf_tr, 5, sig, lam, delta, eps)
```

MSE:

``` r
forecast_bvar <- predict(mod_bvar, h)
(msebvar <- mse(forecast_bvar, etf_te))
#>   EVZCLS   GVZCLS   OVXCLS VXEEMCLS VXEWZCLS VXFXICLS VXGDXCLS VXSLVCLS 
#>     1.36     2.45    59.34     4.23     2.37     3.05     5.75     8.15 
#> VXXLECLS 
#>    42.36
```

### BVHAR

Minnesota-v1:

``` r
mod_bvhar_v1 <- bvhar_minnesota(etf_tr, sigma = sig, lambda = lam, delta = delta, eps = eps)
```

MSE:

``` r
forecast_bvhar_v1 <- predict(mod_bvhar_v1, h)
(msebvhar_v1 <- mse(forecast_bvhar_v1, etf_te))
#>   EVZCLS   GVZCLS   OVXCLS VXEEMCLS VXEWZCLS VXFXICLS VXGDXCLS VXSLVCLS 
#>     1.73     3.24    60.58     3.40     3.78     3.66     6.01     6.89 
#> VXXLECLS 
#>    39.26
```

Minnesota-v2:

``` r
day <- rep(.1, ncol(etf_vix))
week <- rep(.1, ncol(etf_vix))
month <- rep(.1, ncol(etf_vix))
#-------------------------------
mod_bvhar_v2 <- bvhar_minnesota(
  etf_tr, 
  type = "VHAR", 
  sigma = sig, 
  lambda = lam, 
  daily = day, 
  weekly = week, 
  monthly = month, 
  eps = eps
)
```

``` r
forecast_bvhar_v2 <- predict(mod_bvhar_v2, h)
(msebvhar_v2 <- mse(forecast_bvhar_v2, etf_te))
#>   EVZCLS   GVZCLS   OVXCLS VXEEMCLS VXEWZCLS VXFXICLS VXGDXCLS VXSLVCLS 
#>     1.74     2.97    49.28     3.00     5.78     5.86     6.98     6.41 
#> VXXLECLS 
#>    36.94
```

## Compare Models

### Layers

``` r
autoplot(forecast_var, x_cut = 750, ci_alpha = .5) +
  autolayer(forecast_bvhar_v2, ci_alpha = .3)
```

<img src="man/figures/README-unnamed-chunk-16-1.png" width="70%" style="display: block; margin: auto;" />

### Erros

``` r
list(
  forecast_var,
  forecast_vhar,
  forecast_bvar,
  forecast_bvhar_v2
) %>% 
  plot_loss(y = etf_te) +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = -45, vjust = -1))
```

<img src="man/figures/README-msefig-1.png" width="70%" style="display: block; margin: auto;" />

``` r
# VAR---------------
mean(msevar)
#> [1] 15.4
# VHAR--------------
mean(msevhar)
#> [1] 17.4
# BVAR--------------
mean(msebvar)
#> [1] 14.3
# BVHAR-------------
mean(msebvhar_v1)
#> [1] 14.3
mean(msebvhar_v2)
#> [1] 13.2
```

## Code of Conduct

Please note that the bvhar project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
