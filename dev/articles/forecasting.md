# Forecasting

``` r
library(bvhar)
```

## Simulation

Given VAR coefficient and VHAR coefficient each,

- `sim_var(num_sim, num_burn, var_coef, var_lag, sig_error, init)`
  generates VAR process
- `sim_vhar(num_sim, num_burn, vhar_coef, sig_error, init)` generates
  VHAR process

We use coefficient matrix estimated by VAR(5) in introduction vignette.

Consider

``` r
coef(ex_fit)
#>              GVZCLS   OVXCLS    EVZCLS VXFXICLS
#> GVZCLS_1    0.93302 -0.02332 -0.007712 -0.03853
#> OVXCLS_1    0.05429  1.00399  0.009806  0.01062
#> EVZCLS_1    0.06794 -0.13900  0.983825  0.07783
#> VXFXICLS_1 -0.03399  0.03404  0.020719  0.93350
#> GVZCLS_2   -0.07831  0.08753  0.019302  0.08939
#> OVXCLS_2   -0.04770  0.01480  0.003888  0.04392
#> EVZCLS_2    0.08082  0.26704 -0.110017 -0.07163
#> VXFXICLS_2  0.05465 -0.12154 -0.040349  0.04012
#> GVZCLS_3    0.04332 -0.02459 -0.011041 -0.02556
#> OVXCLS_3   -0.00594 -0.09550  0.006638 -0.04981
#> EVZCLS_3   -0.02952 -0.04926  0.091056  0.01204
#> VXFXICLS_3 -0.05876 -0.05995  0.003803 -0.02027
#> GVZCLS_4   -0.00845 -0.04490  0.005415 -0.00817
#> OVXCLS_4    0.01070 -0.00383 -0.022806 -0.05557
#> EVZCLS_4   -0.01971 -0.02008 -0.016535  0.08229
#> VXFXICLS_4  0.06139  0.14403  0.019780 -0.10271
#> GVZCLS_5    0.07301  0.01093 -0.010994 -0.01526
#> OVXCLS_5   -0.01658  0.07401  0.007035  0.04297
#> EVZCLS_5   -0.08794 -0.06189  0.021082 -0.02465
#> VXFXICLS_5 -0.01739  0.00169  0.000335  0.09384
#> const       0.57370  0.15256  0.132842  0.87785
ex_fit$covmat
#>          GVZCLS OVXCLS EVZCLS VXFXICLS
#> GVZCLS    1.157  0.403  0.127    0.332
#> OVXCLS    0.403  1.740  0.115    0.438
#> EVZCLS    0.127  0.115  0.144    0.127
#> VXFXICLS  0.332  0.438  0.127    1.028
```

Then

``` r
m <- ncol(ex_fit$coefficients)
# generate VAR(5)-----------------
y <- sim_var(
  num_sim = 1500, 
  num_burn = 100, 
  var_coef = coef(ex_fit), 
  var_lag = 5L, 
  sig_error = ex_fit$covmat, 
  init = matrix(0L, nrow = 5L, ncol = m)
)
# colname: y1, y2, ...------------
colnames(y) <- paste0("y", 1:m)
head(y)
#>        y1   y2   y3   y4
#> [1,] 10.4 23.7 7.93 25.6
#> [2,] 11.5 22.8 8.57 25.6
#> [3,] 17.1 24.3 8.66 29.2
#> [4,] 16.9 24.1 8.48 29.5
#> [5,] 16.5 23.2 8.29 29.2
#> [6,] 16.3 24.2 8.31 28.7
```

``` r
h <- 20
y_eval <- divide_ts(y, h)
y_train <- y_eval$train # train
y_test <- y_eval$test # test
```

## Fitting Models

### VAR(5) and VHAR

``` r
# VAR(5)
model_var <- var_lm(y_train, 5)
# VHAR
model_vhar <- vhar_lm(y_train)
```

### BVAR(5)

Minnesota prior

``` r
# hyper parameters---------------------------
y_sig <- apply(y_train, 2, sd) # sigma vector
y_lam <- .2 # lambda
y_delta <- rep(.2, m) # delta vector (0 vector since RV stationary)
eps <- 1e-04 # very small number
spec_bvar <- set_bvar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvar <- bvar_minnesota(y_train, p = 5, bayes_spec = spec_bvar)
```

### BVHAR

BVHAR-S

``` r
spec_bvhar_v1 <- set_bvhar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvhar_v1 <- bvhar_minnesota(y_train, bayes_spec = spec_bvhar_v1)
```

BVHAR-L

``` r
# weights----------------------------------
y_day <- rep(.1, m)
y_week <- rep(.01, m)
y_month <- rep(.01, m)
# spec-------------------------------------
spec_bvhar_v2 <- set_weight_bvhar(
  y_sig,
  y_lam,
  eps,
  y_day,
  y_week,
  y_month
)
# fit--------------------------------------
model_bvhar_v2 <- bvhar_minnesota(y_train, bayes_spec = spec_bvhar_v2)
```

## Splitting

You can forecast using [`predict()`](../reference/predict.md) method
with above objects. You should set the step of the forecasting using
`n_ahead` argument.

In addition, the result of this forecast will return another class
called `predbvhar` to use some methods,

- Plot: [`autoplot.predbvhar()`](../reference/autoplot.predbvhar.md)
- Evaluation: [`mse.predbvhar()`](../reference/mse.md),
  [`mae.predbvhar()`](../reference/mae.md),
  [`mape.predbvhar()`](../reference/mape.md),
  [`mase.predbvhar()`](../reference/mase.md),
  [`mrae.predbvhar()`](../reference/mrae.md),
  [`relmae.predbvhar()`](../reference/relmae.md)
- Relative error: [`rmape.predbvhar()`](../reference/rmape.md),
  [`rmase.predbvhar()`](../reference/rmase.md),
  [`rmase.predbvhar()`](../reference/rmase.md),
  [`rmsfe.predbvhar()`](../reference/rmsfe.md),
  [`rmafe.predbvhar()`](../reference/rmafe.md)

### VAR

``` r
(pred_var <- predict(model_var, n_ahead = h))
#>         y1   y2   y3   y4
#>  [1,] 20.1 26.8 10.7 34.1
#>  [2,] 19.8 26.3 10.6 33.6
#>  [3,] 19.7 26.2 10.6 33.1
#>  [4,] 19.7 26.0 10.6 32.7
#>  [5,] 19.7 26.0 10.6 32.3
#>  [6,] 19.6 26.0 10.6 32.0
#>  [7,] 19.5 25.9 10.6 31.7
#>  [8,] 19.5 25.8 10.5 31.4
#>  [9,] 19.4 25.7 10.5 31.1
#> [10,] 19.4 25.6 10.5 30.9
#> [11,] 19.4 25.5 10.5 30.7
#> [12,] 19.3 25.4 10.4 30.4
#> [13,] 19.3 25.3 10.4 30.2
#> [14,] 19.2 25.2 10.4 30.0
#> [15,] 19.2 25.1 10.4 29.8
#> [16,] 19.2 25.0 10.3 29.7
#> [17,] 19.1 24.9 10.3 29.5
#> [18,] 19.1 24.8 10.3 29.4
#> [19,] 19.1 24.7 10.3 29.2
#> [20,] 19.1 24.5 10.2 29.1
```

``` r
class(pred_var)
#> [1] "predbvhar"
names(pred_var)
#> [1] "process"     "forecast"    "se"          "lower"       "upper"      
#> [6] "lower_joint" "upper_joint" "y"
```

The package provides the evaluation function

- `mse(predbvhar, test)`: MSE
- `mape(predbvhar, test)`: MAPE

``` r
(mse_var <- mse(pred_var, y_test))
#>    y1    y2    y3    y4 
#> 4.924 6.479 0.301 1.749
```

### VHAR

``` r
(pred_vhar <- predict(model_vhar, n_ahead = h))
#>         y1   y2   y3   y4
#>  [1,] 19.9 26.5 10.6 34.0
#>  [2,] 19.8 26.1 10.6 33.5
#>  [3,] 19.7 25.9 10.6 33.1
#>  [4,] 19.7 25.7 10.5 32.7
#>  [5,] 19.6 25.6 10.5 32.3
#>  [6,] 19.6 25.5 10.5 31.9
#>  [7,] 19.6 25.4 10.4 31.5
#>  [8,] 19.6 25.3 10.4 31.2
#>  [9,] 19.6 25.2 10.4 30.9
#> [10,] 19.5 25.1 10.4 30.6
#> [11,] 19.5 25.1 10.4 30.3
#> [12,] 19.5 25.0 10.3 30.0
#> [13,] 19.5 25.0 10.3 29.8
#> [14,] 19.5 24.9 10.3 29.6
#> [15,] 19.5 24.9 10.3 29.4
#> [16,] 19.5 24.9 10.3 29.2
#> [17,] 19.5 24.9 10.3 29.0
#> [18,] 19.5 24.9 10.3 28.9
#> [19,] 19.5 24.8 10.3 28.7
#> [20,] 19.5 24.8 10.3 28.6
```

MSE:

``` r
(mse_vhar <- mse(pred_vhar, y_test))
#>    y1    y2    y3    y4 
#> 4.002 6.965 0.246 2.235
```

### BVAR

``` r
(pred_bvar <- predict(model_bvar, n_ahead = h))
#>              y1       y2       y3       y4
#>  [1,]  2.15e+01 2.15e+01 1.55e+01 2.52e+01
#>  [2,]  2.54e+01 1.96e+01 2.91e+01 2.47e+01
#>  [3,]  3.56e+01 1.99e+01 6.60e+01 2.50e+01
#>  [4,]  6.11e+01 2.32e+01 1.65e+02 2.66e+01
#>  [5,]  1.25e+02 3.34e+01 4.28e+02 3.11e+01
#>  [6,]  2.81e+02 6.12e+01 1.13e+03 4.32e+01
#>  [7,]  6.66e+02 1.35e+02 3.01e+03 7.55e+01
#>  [8,]  1.60e+03 3.31e+02 8.04e+03 1.61e+02
#>  [9,]  3.87e+03 8.51e+02 2.15e+04 3.88e+02
#> [10,]  9.30e+03 2.23e+03 5.74e+04 9.89e+02
#> [11,]  2.22e+04 5.86e+03 1.53e+05 2.58e+03
#> [12,]  5.22e+04 1.55e+04 4.10e+05 6.80e+03
#> [13,]  1.21e+05 4.10e+04 1.10e+06 1.80e+04
#> [14,]  2.74e+05 1.09e+05 2.93e+06 4.77e+04
#> [15,]  6.04e+05 2.88e+05 7.85e+06 1.26e+05
#> [16,]  1.27e+06 7.62e+05 2.10e+07 3.35e+05
#> [17,]  2.50e+06 2.02e+06 5.62e+07 8.90e+05
#> [18,]  4.31e+06 5.36e+06 1.50e+08 2.36e+06
#> [19,]  5.26e+06 1.42e+07 4.03e+08 6.27e+06
#> [20,] -2.51e+06 3.77e+07 1.08e+09 1.66e+07
```

MSE:

``` r
(mse_bvar <- mse(pred_bvar, y_test))
#>       y1       y2       y3       y4 
#> 3.05e+12 8.27e+13 6.76e+16 1.61e+13
```

### BVHAR

#### VAR-type Minnesota

``` r
(pred_bvhar_v1 <- predict(model_bvhar_v1, n_ahead = h))
#>              y1        y2       y3       y4
#>  [1,]  3.10e+01  1.06e+02 4.00e+01 1.25e+02
#>  [2,]  1.10e+02  7.67e+02 2.91e+02 9.72e+02
#>  [3,]  6.44e+02  6.23e+03 2.43e+03 8.91e+03
#>  [4,]  3.94e+03  5.12e+04 2.06e+04 8.35e+04
#>  [5,]  2.00e+04  4.20e+05 1.76e+05 7.86e+05
#>  [6,]  4.37e+04  3.43e+06 1.49e+06 7.42e+06
#>  [7,] -8.25e+05  2.78e+07 1.27e+07 7.01e+07
#>  [8,] -1.84e+07  2.23e+08 1.07e+08 6.64e+08
#>  [9,] -2.64e+08  1.77e+09 9.06e+08 6.30e+09
#> [10,] -3.27e+09  1.39e+10 7.63e+09 5.99e+10
#> [11,] -3.75e+10  1.07e+11 6.39e+10 5.70e+11
#> [12,] -4.11e+11  7.99e+11 5.33e+11 5.43e+12
#> [13,] -4.37e+12  5.77e+12 4.42e+12 5.17e+13
#> [14,] -4.55e+13  3.92e+13 3.63e+13 4.94e+14
#> [15,] -4.66e+14  2.40e+14 2.95e+14 4.72e+15
#> [16,] -4.72e+15  1.15e+15 2.36e+15 4.51e+16
#> [17,] -4.73e+16  1.19e+15 1.86e+16 4.31e+17
#> [18,] -4.71e+17 -7.18e+16 1.42e+17 4.13e+18
#> [19,] -4.65e+18 -1.39e+18 1.04e+18 3.95e+19
#> [20,] -4.58e+19 -1.93e+19 7.11e+18 3.79e+20
```

MSE:

``` r
(mse_bvhar_v1 <- mse(pred_bvhar_v1, y_test))
#>       y1       y2       y3       y4 
#> 1.06e+38 1.88e+37 2.58e+36 7.26e+39
```

#### VHAR-type Minnesota

``` r
(pred_bvhar_v2 <- predict(model_bvhar_v2, n_ahead = h))
#>             y1       y2       y3       y4
#>  [1,] 2.20e+01 1.99e+01 8.02e+00 2.59e+01
#>  [2,] 2.81e+01 1.77e+01 7.29e+00 2.44e+01
#>  [3,] 4.45e+01 1.74e+01 7.17e+00 2.42e+01
#>  [4,] 8.74e+01 1.83e+01 7.37e+00 2.44e+01
#>  [5,] 2.00e+02 2.12e+01 8.09e+00 2.54e+01
#>  [6,] 4.94e+02 2.92e+01 1.01e+01 2.79e+01
#>  [7,] 1.27e+03 5.06e+01 1.55e+01 3.49e+01
#>  [8,] 3.30e+03 1.07e+02 2.98e+01 5.31e+01
#>  [9,] 8.65e+03 2.55e+02 6.73e+01 1.01e+02
#> [10,] 2.27e+04 6.44e+02 1.66e+02 2.27e+02
#> [11,] 5.95e+04 1.67e+03 4.25e+02 5.59e+02
#> [12,] 1.56e+05 4.35e+03 1.11e+03 1.43e+03
#> [13,] 4.11e+05 1.14e+04 2.89e+03 3.72e+03
#> [14,] 1.08e+06 2.99e+04 7.59e+03 9.73e+03
#> [15,] 2.83e+06 7.86e+04 1.99e+04 2.55e+04
#> [16,] 7.44e+06 2.07e+05 5.23e+04 6.70e+04
#> [17,] 1.96e+07 5.42e+05 1.37e+05 1.76e+05
#> [18,] 5.14e+07 1.42e+06 3.61e+05 4.62e+05
#> [19,] 1.35e+08 3.74e+06 9.48e+05 1.21e+06
#> [20,] 3.54e+08 9.83e+06 2.49e+06 3.19e+06
```

MSE:

``` r
(mse_bvhar_v2 <- mse(pred_bvhar_v2, y_test))
#>       y1       y2       y3       y4 
#> 7.34e+15 5.65e+12 3.62e+11 5.94e+11
```

### Compare

#### Region

`autoplot(predbvhar)` and `autolayer(predbvhar)` draws the results of
the forecasting.

``` r
autoplot(pred_var, x_cut = 1470, ci_alpha = .7, type = "wrap") +
  autolayer(pred_vhar, ci_alpha = .5) +
  autolayer(pred_bvar, ci_alpha = .4) +
  autolayer(pred_bvhar_v1, ci_alpha = .2) +
  autolayer(pred_bvhar_v2, ci_alpha = .1) +
  geom_eval(y_test, colour = "#000000", alpha = .5)
#> Warning: `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
```

![](forecasting_files/figure-html/predplot-1.png)

#### Error

Mean of MSE

``` r
list(
  VAR = mse_var,
  VHAR = mse_vhar,
  BVAR = mse_bvar,
  BVHAR1 = mse_bvhar_v1,
  BVHAR2 = mse_bvhar_v2
) |> 
  lapply(mean) |> 
  unlist() |> 
  sort()
#>     VHAR      VAR   BVHAR2     BVAR   BVHAR1 
#> 3.36e+00 3.36e+00 1.84e+15 1.69e+16 1.85e+39
```

For each variable, we can see the error with plot.

``` r
list(
  pred_var,
  pred_vhar,
  pred_bvar,
  pred_bvhar_v1,
  pred_bvhar_v2
) |> 
  gg_loss(y = y_test, "mse")
#> Warning: `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
```

![](forecasting_files/figure-html/evalplot-1.png)

Relative MAPE (MAPE), benchmark model: VAR

``` r
list(
  VAR = pred_var,
  VHAR = pred_vhar,
  BVAR = pred_bvar,
  BVHAR1 = pred_bvhar_v1,
  BVHAR2 = pred_bvhar_v2
) |> 
  lapply(rmape, pred_bench = pred_var, y = y_test) |> 
  unlist()
#>      VAR     VHAR     BVAR   BVHAR1   BVHAR2 
#> 1.00e+00 9.66e-01 3.46e+07 3.56e+18 5.47e+06
```

## Out-of-Sample Forecasting

In time series research, out-of-sample forecasting plays a key role. So,
we provide out-of-sample forecasting function based on

- Rolling window: `forecast_roll(object, n_ahead, y_test)`
- Expanding window: `forecast_expand(object, n_ahead, y_test)`

### Rolling windows

`forecast_roll(object, n_ahead, y_test)` conducts h \>= 1 step rolling
windows forecasting.

It fixes window size and moves the window. The window is the training
set. In this package, we set *window size = original input data*.

Iterating the step

1.  The model is fitted in the training set.
2.  With the fitted model, researcher should forecast the next h \>= 1
    step ahead. The longest forecast horizon is `num_test - h + 1`.
3.  After this window, move the window and do the same process.
4.  Get forecasted values until possible (longest forecast horizon).

5-step out-of-sample:

``` r
(var_roll <- forecast_roll(model_var, 5, y_test))
#>         y1   y2    y3   y4
#>  [1,] 19.7 26.0 10.58 32.3
#>  [2,] 18.6 25.1 10.08 31.3
#>  [3,] 18.4 24.3  9.84 30.9
#>  [4,] 18.1 24.5  9.68 30.5
#>  [5,] 19.2 25.5  9.99 31.3
#>  [6,] 19.2 26.8 10.23 30.9
#>  [7,] 19.3 27.3 10.13 30.1
#>  [8,] 20.4 28.2 10.38 29.5
#>  [9,] 20.1 28.3 10.30 30.3
#> [10,] 19.9 27.0 10.02 29.3
#> [11,] 19.6 26.7 10.05 29.5
#> [12,] 19.6 25.5  9.67 30.3
#> [13,] 20.8 26.6 10.14 29.8
#> [14,] 20.8 26.2 10.06 28.9
#> [15,] 20.6 25.6  9.91 28.9
#> [16,] 19.8 26.6 10.22 28.4
```

Denote that the nrow is longest forecast horizon.

``` r
class(var_roll)
#> [1] "predbvhar_roll" "bvharcv"
names(var_roll)
#> [1] "process"  "forecast" "eval_id"  "y"
```

To apply the same evaluation methods, a class named `bvharcv` has been
defined. You can use the functions above.

``` r
vhar_roll <- forecast_roll(model_vhar, 5, y_test)
bvar_roll <- forecast_roll(model_bvar, 5, y_test)
bvhar_roll_v1 <- forecast_roll(model_bvhar_v1, 5, y_test)
bvhar_roll_v2 <- forecast_roll(model_bvhar_v2, 5, y_test)
```

Relative MAPE, benchmark model: VAR

``` r
list(
  VAR = var_roll,
  VHAR = vhar_roll,
  BVAR = bvar_roll,
  BVHAR1 = bvhar_roll_v1,
  BVHAR2 = bvhar_roll_v2
) |> 
  lapply(rmape, pred_bench = var_roll, y = y_test) |> 
  unlist()
#>      VAR     VHAR     BVAR   BVHAR1   BVHAR2 
#> 1.00e+00 9.87e-01 1.44e+04 2.53e+05 2.27e+04
```

### Expanding Windows

`forecast_expand(object, n_ahead, y_test)` conducts h \>= 1 step
expanding window forecasting.

Different with rolling windows, expanding windows method fixes the
starting point. The other is same.

``` r
(var_expand <- forecast_expand(model_var, 5, y_test))
#>         y1   y2    y3   y4
#>  [1,] 19.7 26.0 10.58 32.3
#>  [2,] 18.6 25.1 10.09 31.3
#>  [3,] 18.4 24.4  9.84 30.9
#>  [4,] 18.1 24.5  9.68 30.4
#>  [5,] 19.2 25.6  9.99 31.3
#>  [6,] 19.2 26.8 10.25 30.9
#>  [7,] 19.3 27.3 10.15 30.2
#>  [8,] 20.4 28.2 10.39 29.6
#>  [9,] 20.0 28.3 10.31 30.3
#> [10,] 19.9 27.0 10.02 29.3
#> [11,] 19.6 26.7 10.06 29.5
#> [12,] 19.5 25.5  9.67 30.2
#> [13,] 20.7 26.6 10.13 29.8
#> [14,] 20.7 26.2 10.06 28.9
#> [15,] 20.5 25.6  9.92 28.9
#> [16,] 19.8 26.6 10.22 28.4
```

The class is `bvharcv`.

``` r
class(var_expand)
#> [1] "predbvhar_expand" "bvharcv"
names(var_expand)
#> [1] "process"  "forecast" "eval_id"  "y"
```

``` r
vhar_expand <- forecast_expand(model_vhar, 5, y_test)
bvar_expand <- forecast_expand(model_bvar, 5, y_test)
bvhar_expand_v1 <- forecast_expand(model_bvhar_v1, 5, y_test)
bvhar_expand_v2 <- forecast_expand(model_bvhar_v2, 5, y_test)
```

Relative MAPE, benchmark model: VAR

``` r
list(
  VAR = var_expand,
  VHAR = vhar_expand,
  BVAR = bvar_expand,
  BVHAR1 = bvhar_expand_v1,
  BVHAR2 = bvhar_expand_v2
) |> 
  lapply(rmape, pred_bench = var_expand, y = y_test) |> 
  unlist()
#>      VAR     VHAR     BVAR   BVHAR1   BVHAR2 
#> 1.00e+00 9.80e-01 8.25e+02 2.33e+05 7.41e+04
```
