
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bvhar <img src='man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/ygeunkim/bvhar/workflows/R-CMD-check/badge.svg)](https://github.com/ygeunkim/bvhar/actions)
[![Codecov test
coverage](https://codecov.io/gh/ygeunkim/bvhar/branch/master/graph/badge.svg?token=umidjitjiK)](https://codecov.io/gh/ygeunkim/bvhar?branch=master)
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

| Model |      function       |     S3     |
|:-----:|:-------------------:|:----------:|
|  VAR  |     `var_lm()`      |  `varlse`  |
| VHAR  |     `vhar_lm()`     | `vharlse`  |
| BVAR  | `bvar_minnesota()`  |  `bvarmn`  |
| BVAR  |    `bvar_flat()`    | `bvarflat` |
| BVHAR | `bvhar_minnesota()` | `bvharmn`  |

As the other analyzer tools uses S3 such as `lm`, this package use
methods `coef`, `predict`, etc. This readme document shows forecasting
procedure briefly. Details about each function are in vignettes and help
documents.

h-step ahead forecasting:

``` r
h <- 19
etf_split <- divide_ts(etf_vix, h) # Try ?divide_ts
etf_tr <- etf_split$train
etf_te <- etf_split$test
```

### VAR

VAR(5):

``` r
mod_var <- var_lm(y = etf_tr, p = 5)
```

Forecasting:

``` r
forecast_var <- predict(mod_var, h)
```

MSE:

``` r
(msevar <- mse(forecast_var, etf_te))
#>   GVZCLS   OVXCLS VXFXICLS VXEEMCLS VXSLVCLS   EVZCLS VXXLECLS VXGDXCLS 
#>    5.381   14.689    2.838    9.451   10.078    0.654   22.436    9.992 
#> VXEWZCLS 
#>   10.647
```

### VHAR

``` r
mod_vhar <- vhar_lm(y = etf_tr)
```

MSE:

``` r
forecast_vhar <- predict(mod_vhar, h)
(msevhar <- mse(forecast_vhar, etf_te))
#>   GVZCLS   OVXCLS VXFXICLS VXEEMCLS VXSLVCLS   EVZCLS VXXLECLS VXGDXCLS 
#>     6.15     2.49     1.52     1.58    10.55     1.35     8.79     4.43 
#> VXEWZCLS 
#>     3.84
```

### BVAR

Minnesota prior:

``` r
lam <- .3
delta <- rep(1, ncol(etf_vix)) # litterman
sig <- apply(etf_tr, 2, sd)
eps <- 1e-04
(bvar_spec <- set_bvar(sig, lam, delta, eps))
#> Model Specification for BVAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: Minnesota
#> # Type '?bvar_minnesota' in the console for some help.
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS  VXFXICLS  VXEEMCLS  VXSLVCLS    EVZCLS  VXXLECLS  VXGDXCLS  
#>     3.77     10.63      3.81      4.39      5.99      2.27      4.88      7.45  
#> VXEWZCLS  
#>     7.03  
#> 
#> Setting for 'lambda':
#> [1]  0.3
#> 
#> Setting for 'delta':
#> [1]  1  1  1  1  1  1  1  1  1
#> 
#> Setting for 'eps':
#> [1]  1e-04
```

``` r
mod_bvar <- bvar_minnesota(y = etf_tr, p = 5, bayes_spec = bvar_spec)
```

MSE:

``` r
forecast_bvar <- predict(mod_bvar, h)
(msebvar <- mse(forecast_bvar, etf_te))
#>   GVZCLS   OVXCLS VXFXICLS VXEEMCLS VXSLVCLS   EVZCLS VXXLECLS VXGDXCLS 
#>    4.651   13.248    1.845   10.356    9.894    0.667   21.040    6.262 
#> VXEWZCLS 
#>    8.864
```

### BVHAR

BVHAR-S:

``` r
(bvhar_spec_v1 <- set_bvhar(sig, lam, delta, eps))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VAR
#> # Type '?bvhar_minnesota' in the console for some help.
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS  VXFXICLS  VXEEMCLS  VXSLVCLS    EVZCLS  VXXLECLS  VXGDXCLS  
#>     3.77     10.63      3.81      4.39      5.99      2.27      4.88      7.45  
#> VXEWZCLS  
#>     7.03  
#> 
#> Setting for 'lambda':
#> [1]  0.3
#> 
#> Setting for 'delta':
#> [1]  1  1  1  1  1  1  1  1  1
#> 
#> Setting for 'eps':
#> [1]  1e-04
```

``` r
mod_bvhar_v1 <- bvhar_minnesota(y = etf_tr, bayes_spec = bvhar_spec_v1)
```

MSE:

``` r
forecast_bvhar_v1 <- predict(mod_bvhar_v1, h)
(msebvhar_v1 <- mse(forecast_bvhar_v1, etf_te))
#>   GVZCLS   OVXCLS VXFXICLS VXEEMCLS VXSLVCLS   EVZCLS VXXLECLS VXGDXCLS 
#>    3.199    6.067    1.471    5.142    5.946    0.878   12.165    2.553 
#> VXEWZCLS 
#>    6.462
```

BVHAR-L:

``` r
day <- rep(.1, ncol(etf_vix))
week <- rep(.1, ncol(etf_vix))
month <- rep(.1, ncol(etf_vix))
#----------------------------------
(bvhar_spec_v2 <- set_weight_bvhar(sig, lam, eps, day, week, month))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VHAR
#> # Type '?bvhar_minnesota' in the console for some help.
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS  VXFXICLS  VXEEMCLS  VXSLVCLS    EVZCLS  VXXLECLS  VXGDXCLS  
#>     3.77     10.63      3.81      4.39      5.99      2.27      4.88      7.45  
#> VXEWZCLS  
#>     7.03  
#> 
#> Setting for 'lambda':
#> [1]  0.3
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'daily':
#> [1]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
#> 
#> Setting for 'weekly':
#> [1]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
#> 
#> Setting for 'monthly':
#> [1]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
```

``` r
mod_bvhar_v2 <- bvhar_minnesota(y = etf_tr, bayes_spec = bvhar_spec_v2)
```

MSE:

``` r
forecast_bvhar_v2 <- predict(mod_bvhar_v2, h)
(msebvhar_v2 <- mse(forecast_bvhar_v2, etf_te))
#>   GVZCLS   OVXCLS VXFXICLS VXEEMCLS VXSLVCLS   EVZCLS VXXLECLS VXGDXCLS 
#>     3.63     3.85     1.64     5.12     5.75     1.08    13.60     2.58 
#> VXEWZCLS 
#>     5.54
```

## Compare Models

### Layers

``` r
autoplot(forecast_var, x_cut = 870, ci_alpha = .7, type = "wrap") +
  autolayer(forecast_vhar, ci_alpha = .6) +
  autolayer(forecast_bvar, ci_alpha = .4) +
  autolayer(forecast_bvhar_v1, ci_alpha = .2) +
  autolayer(forecast_bvhar_v2, ci_alpha = .1)
```

<img src="man/figures/README-predfig-1.png" width="70%" style="display: block; margin: auto;" />

### Erros

``` r
list(
  forecast_var,
  forecast_vhar,
  forecast_bvar,
  forecast_bvhar_v2
) %>% 
  gg_loss(y = etf_te, mean_line = TRUE, mean_param = list(alpha = .5)) +
  ggplot2::theme_minimal() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = -45, vjust = -1))
```

<img src="man/figures/README-msefig-1.png" width="70%" style="display: block; margin: auto;" />

## Code of Conduct

Please note that the bvhar project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
