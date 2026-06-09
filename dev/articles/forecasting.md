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
#>         y1   y2   y3   y4
#>  [1,] 19.8 22.2 8.64 27.8
#>  [2,] 19.4 19.8 7.87 25.7
#>  [3,] 19.1 18.4 7.43 24.8
#>  [4,] 18.8 17.4 7.15 24.3
#>  [5,] 18.7 16.8 6.97 24.0
#>  [6,] 18.6 16.4 6.84 23.8
#>  [7,] 18.5 16.1 6.77 23.7
#>  [8,] 18.4 16.0 6.72 23.6
#>  [9,] 18.4 15.8 6.69 23.6
#> [10,] 18.4 15.8 6.67 23.5
#> [11,] 18.4 15.7 6.66 23.5
#> [12,] 18.4 15.7 6.65 23.5
#> [13,] 18.4 15.7 6.65 23.5
#> [14,] 18.4 15.7 6.65 23.5
#> [15,] 18.4 15.7 6.64 23.5
#> [16,] 18.4 15.7 6.64 23.5
#> [17,] 18.4 15.7 6.64 23.5
#> [18,] 18.4 15.7 6.64 23.5
#> [19,] 18.4 15.7 6.64 23.5
#> [20,] 18.4 15.7 6.64 23.5
```

MSE:

``` r

(mse_bvar <- mse(pred_bvar, y_test))
#>     y1     y2     y3     y4 
#>   7.81 121.49  11.70  54.33
```

### BVHAR

#### VAR-type Minnesota

``` r

(pred_bvhar_v1 <- predict(model_bvhar_v1, n_ahead = h))
#>             y1       y2       y3       y4
#>  [1,] 3.17e+01 1.84e+01 4.18e+01 2.47e+01
#>  [2,] 1.46e+02 1.76e+01 3.52e+02 2.41e+01
#>  [3,] 1.25e+03 2.65e+01 3.41e+03 2.74e+01
#>  [4,] 1.19e+04 1.18e+02 3.37e+04 6.09e+01
#>  [5,] 1.15e+05 1.02e+03 3.32e+05 3.90e+02
#>  [6,] 1.11e+06 9.86e+03 3.28e+06 3.63e+03
#>  [7,] 1.08e+07 9.68e+04 3.24e+07 3.55e+04
#>  [8,] 1.04e+08 9.51e+05 3.20e+08 3.49e+05
#>  [9,] 1.00e+09 9.35e+06 3.16e+09 3.43e+06
#> [10,] 9.67e+09 9.18e+07 3.12e+10 3.37e+07
#> [11,] 9.33e+10 9.03e+08 3.09e+11 3.31e+08
#> [12,] 9.00e+11 8.87e+09 3.05e+12 3.26e+09
#> [13,] 8.67e+12 8.72e+10 3.01e+13 3.20e+10
#> [14,] 8.35e+13 8.57e+11 2.97e+14 3.15e+11
#> [15,] 8.04e+14 8.42e+12 2.94e+15 3.10e+12
#> [16,] 7.74e+15 8.28e+13 2.90e+16 3.04e+13
#> [17,] 7.45e+16 8.14e+14 2.87e+17 2.99e+14
#> [18,] 7.16e+17 8.00e+15 2.83e+18 2.94e+15
#> [19,] 6.88e+18 7.87e+16 2.80e+19 2.90e+16
#> [20,] 6.61e+19 7.73e+17 2.77e+20 2.85e+17
```

MSE:

``` r

(mse_bvhar_v1 <- mse(pred_bvhar_v1, y_test))
#>       y1       y2       y3       y4 
#> 2.21e+38 3.02e+34 3.86e+39 4.10e+33
```

#### VHAR-type Minnesota

``` r

(pred_bvhar_v2 <- predict(model_bvhar_v2, n_ahead = h))
#>              y1       y2       y3       y4
#>  [1,]  3.72e+01 1.87e+01 7.56e+00 1.95e+02
#>  [2,]  2.05e+02 2.06e+01 8.49e+00 1.90e+03
#>  [3,]  1.82e+03 5.94e+01 2.41e+01 2.01e+04
#>  [4,]  1.71e+04 4.76e+02 1.91e+02 2.15e+05
#>  [5,]  1.59e+05 4.89e+03 1.97e+03 2.30e+06
#>  [6,]  1.44e+06 5.17e+04 2.09e+04 2.46e+07
#>  [7,]  1.25e+07 5.49e+05 2.22e+05 2.63e+08
#>  [8,]  1.02e+08 5.82e+06 2.37e+06 2.82e+09
#>  [9,]  7.44e+08 6.16e+07 2.52e+07 3.02e+10
#> [10,]  4.15e+09 6.53e+08 2.68e+08 3.23e+11
#> [11,]  2.75e+09 6.92e+09 2.85e+09 3.46e+12
#> [12,] -4.28e+11 7.33e+10 3.03e+10 3.71e+13
#> [13,] -9.59e+12 7.76e+11 3.22e+11 3.97e+14
#> [14,] -1.58e+14 8.22e+12 3.42e+12 4.25e+15
#> [15,] -2.29e+15 8.70e+13 3.64e+13 4.55e+16
#> [16,] -3.11e+16 9.20e+14 3.87e+14 4.88e+17
#> [17,] -4.06e+17 9.74e+15 4.11e+15 5.23e+18
#> [18,] -5.14e+18 1.03e+17 4.37e+16 5.60e+19
#> [19,] -6.37e+19 1.09e+18 4.64e+17 6.00e+20
#> [20,] -7.78e+20 1.15e+19 4.93e+18 6.44e+21
```

MSE:

``` r

(mse_bvhar_v2 <- mse(pred_bvhar_v2, y_test))
#>       y1       y2       y3       y4 
#> 3.04e+40 6.67e+36 1.23e+36 2.09e+42
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
#>     VHAR      VAR     BVAR   BVHAR1   BVHAR2 
#> 3.36e+00 3.36e+00 4.88e+01 1.02e+39 5.30e+41
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
#> 1.00e+00 9.66e-01 4.59e+00 6.61e+18 5.45e+19
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
#> 1.00e+00 9.87e-01 2.00e+04 1.45e+05 2.49e+05
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
#> 1.00e+00 9.80e-01 2.55e+04 1.42e+05 2.39e+05
```
