# Introduction to bvhar

``` r
library(bvhar)
```

## Data

Looking at VAR and VHAR, you can learn how the models work and how to
perform this package.

### ETF Dataset

This package includes some datasets. Among them, we try CBOE ETF
volatility index (`etf_vix`). Since this is just an example, we
arbitrarily extract a small number of variables: *Gold, crude oil, euro
currency, and china ETF*.

``` r
var_idx <- c("GVZCLS", "OVXCLS", "EVZCLS", "VXFXICLS")
etf <- 
  etf_vix |> 
  dplyr::select(dplyr::all_of(var_idx))
etf
#> # A tibble: 905 × 4
#>    GVZCLS OVXCLS EVZCLS VXFXICLS
#>     <dbl>  <dbl>  <dbl>    <dbl>
#>  1   21.5   36.5   13.2     30.2
#>  2   21.5   35.4   12.6     28.9
#>  3   22.3   35.5   13.1     29.1
#>  4   21.6   36.6   12.8     28.5
#>  5   21.2   35.6   13.3     29.5
#>  6   21.4   34.8   13.2     29.1
#>  7   21.6   34.0   13.2     28.7
#>  8   21.1   32.6   12.8     28.0
#>  9   20.3   33.5   12.7     28.9
#> 10   19.6   33.4   12.4     28.0
#> # ℹ 895 more rows
```

### h-step ahead forecasting

For evaluation, split the data. The last `19` observations will be test
set. [`divide_ts()`](../reference/divide_ts.md) function splits the time
series into train-test set.

In the other vignette, we provide how to perform out-of-sample
forecasting.

``` r
h <- 19
etf_eval <- divide_ts(etf, h) # Try ?divide_ts
etf_train <- etf_eval$train # train
etf_test <- etf_eval$test # test
# dimension---------
m <- ncol(etf)
```

- T: Total number of observation
- p: VAR lag
- m: Dimension of variable
- n = T - p
- k = m \* p + 1 if constant term, k = m \* p without constant term

## Models

### VAR

This package indentifies VAR(p) model by

$$\mathbf{Y}_{t} = \mathbf{c} + {\mathbf{β}}_{1}\mathbf{Y}_{t - 1} + \ldots + {\mathbf{β}}_{p} + \mathbf{Y}_{t - p} + {\mathbf{ϵ}}_{t}$$

where ${\mathbf{ϵ}}_{t} \sim N\left( \mathbf{0}_{k},\Sigma_{e} \right)$

``` r
var_lag <- 5
```

The package perform VAR(p = 5) based on

$$Y_{0} = X_{0}A + Z$$

where

$$Y_{0} = \begin{bmatrix}
\mathbf{y}_{p + 1}^{T} \\
\mathbf{y}_{p + 2}^{T} \\
\vdots \\
\mathbf{y}_{n}^{T}
\end{bmatrix}_{s \times m} \equiv Y_{p + 1} \in {\mathbb{R}}^{s \times m}$$

by `build_y0()`

and

$$X_{0} = \begin{bmatrix}
\mathbf{y}_{p}^{T} & \cdots & \mathbf{y}_{1}^{T} & 1 \\
\mathbf{y}_{p + 1}^{T} & \cdots & \mathbf{y}_{2}^{T} & 1 \\
\vdots & \vdots & \cdots & \vdots \\
\mathbf{y}_{T - 1}^{T} & \cdots & \mathbf{y}_{T - p}^{T} & 1
\end{bmatrix}_{s \times k} = \begin{bmatrix}
Y_{p} & Y_{p - 1} & \cdots & \mathbf{1}_{T - p}
\end{bmatrix} \in {\mathbb{R}}^{s \times k}$$

by `build_design()`. Coefficient matrix is the form of

$$A = \begin{bmatrix}
A_{1}^{T} \\
\vdots \\
A_{p}^{T} \\
\mathbf{c}^{T}
\end{bmatrix} \in {\mathbb{R}}^{k \times m}$$

This form also corresponds to the other model. Use `var_lm(y, p)` to
model VAR(p). You can specify `type = "none"` to get model without
constant term.

``` r
(fit_var <- var_lm(y = etf_train, p = var_lag))
#> Call:
#> var_lm(y = etf_train, p = var_lag)
#> 
#> VAR(5) Estimation using least squares
#> ====================================================
#> 
#> LSE for A1:
#>           GVZCLS_1  OVXCLS_1  EVZCLS_1  VXFXICLS_1
#> GVZCLS     0.93290    0.0545    0.0659     -0.0346
#> OVXCLS    -0.02367    1.0047   -0.1447      0.0324
#> EVZCLS    -0.00789    0.0102    0.9810      0.0199
#> VXFXICLS  -0.03868    0.0109    0.0754      0.9328
#> 
#> 
#> LSE for A2:
#>           GVZCLS_2  OVXCLS_2  EVZCLS_2  VXFXICLS_2
#> GVZCLS     -0.0781  -0.04865    0.0829      0.0561
#> OVXCLS      0.0880   0.01207    0.2729     -0.1173
#> EVZCLS      0.0195   0.00255   -0.1071     -0.0383
#> VXFXICLS    0.0896   0.04278   -0.0691      0.0419
#> 
#> 
#> LSE for A3:
#>           GVZCLS_3  OVXCLS_3  EVZCLS_3  VXFXICLS_3
#> GVZCLS      0.0424  -0.00452  -0.03245    -0.05967
#> OVXCLS     -0.0272  -0.09144  -0.05764    -0.06255
#> EVZCLS     -0.0123   0.00864   0.08693     0.00252
#> VXFXICLS   -0.0266  -0.04810   0.00851    -0.02137
#> 
#> 
#> LSE for A4:
#>           GVZCLS_4  OVXCLS_4  EVZCLS_4  VXFXICLS_4
#> GVZCLS    -0.00793   0.01072  -0.01513      0.0616
#> OVXCLS    -0.04343  -0.00377  -0.00694      0.1445
#> EVZCLS     0.00614  -0.02278  -0.01007      0.0200
#> VXFXICLS  -0.00755  -0.05555   0.08783     -0.1025
#> 
#> 
#> LSE for A5:
#>           GVZCLS_5  OVXCLS_5  EVZCLS_5  VXFXICLS_5
#> GVZCLS      0.0728  -0.01745   -0.0886   -0.017273
#> OVXCLS      0.0104   0.07151   -0.0637    0.002018
#> EVZCLS     -0.0113   0.00581    0.0202    0.000498
#> VXFXICLS   -0.0155   0.04192   -0.0254    0.093984
#> 
#> 
#> LSE for constant:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    0.571     0.145     0.129     0.875  
#> 
#> 
#> --------------------------------------------------
#> *_j of the Coefficient matrix: corresponding to the j-th VAR lag
```

The package provide `S3` object.

``` r
# class---------------
class(fit_var)
#> [1] "varlse"   "olsmod"   "bvharmod"
# inheritance---------
is.varlse(fit_var)
#> [1] TRUE
# names---------------
names(fit_var)
#>  [1] "coefficients"  "fitted.values" "residuals"     "covmat"       
#>  [5] "df"            "m"             "obs"           "y0"           
#>  [9] "p"             "totobs"        "process"       "type"         
#> [13] "design"        "y"             "method"        "call"
```

### VHAR

Consider Vector HAR (VHAR) model.

$$\mathbf{Y}_{t} = \mathbf{c} + \Phi^{(d)} + \mathbf{Y}_{t - 1} + \Phi^{(w)}\mathbf{Y}_{t - 1}^{(w)} + \Phi^{(m)}\mathbf{Y}_{t - 1}^{(m)} + {\mathbf{ϵ}}_{t}$$

where $\mathbf{Y}_{t}$ is daily RV and

$$\mathbf{Y}_{t}^{(w)} = \frac{1}{5}\left( \mathbf{Y}_{t} + \cdots + \mathbf{Y}_{t - 4} \right)$$

is weekly RV

and

$$\mathbf{Y}_{t}^{(m)} = \frac{1}{22}\left( \mathbf{Y}_{t} + \cdots + \mathbf{Y}_{t - 21} \right)$$

is monthly RV. This model can be expressed by

$$Y_{0} = X_{1}\Phi + Z$$

where

$$\Phi = \begin{bmatrix}
\Phi^{{(d)}T} \\
\Phi^{{(w)}T} \\
\Phi^{{(m)}T} \\
\mathbf{c}^{T}
\end{bmatrix} \in {\mathbb{R}}^{{(3m + 1)} \times m}$$

Let $T$ be

$${\mathbb{C}}_{0}: = \begin{bmatrix}
1 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
{1/5} & {1/5} & \cdots & {1/5} & 0 & \cdots & 0 \\
{1/22} & {1/22} & \cdots & {1/22} & {1/22} & \cdots & {1/22}
\end{bmatrix} \otimes I_{m} \in {\mathbb{R}}^{3m \times 22m}$$

and let ${\mathbb{C}}_{HAR}$ be

$${\mathbb{C}}_{HAR}: = \begin{bmatrix}
T & \mathbf{0}_{3m} \\
\mathbf{0}_{3m}^{T} & 1
\end{bmatrix} \in {\mathbb{R}}^{{(3m + 1)} \times {(22m + 1)}}$$

Then for $X_{0}$ in VAR(p),

$$X_{1} = X_{0}{\mathbb{C}}_{HAR}^{T} = \begin{bmatrix}
\mathbf{y}_{22}^{T} & \mathbf{y}_{22}^{{(w)}T} & \mathbf{y}_{22}^{{(m)}T} & 1 \\
\mathbf{y}_{23}^{T} & \mathbf{y}_{23}^{{(w)}T} & \mathbf{y}_{23}^{{(m)}T} & 1 \\
\vdots & \vdots & \vdots & \vdots \\
\mathbf{y}_{T - 1}^{T} & \mathbf{y}_{T - 1}^{{(w)}T} & \mathbf{y}_{T - 1}^{{(m)}T} & 1
\end{bmatrix} \in {\mathbb{R}}^{s \times {(3m + 1)}}$$

This package fits VHAR by scaling VAR(p) using ${\mathbb{C}}_{HAR}$
(`scale_har(m, week = 5, month = 22)`). Use `vhar_lm(y)` to fit VHAR.
You can specify `type = "none"` to get model without constant term.

``` r
(fit_har <- vhar_lm(y = etf_train))
#> Call:
#> vhar_lm(y = etf_train)
#> 
#> VHAR Estimation====================================================
#> 
#> LSE for day:
#>           GVZCLS_day  OVXCLS_day  EVZCLS_day  VXFXICLS_day
#> GVZCLS       0.87561      0.0447      0.1623      -0.03772
#> OVXCLS       0.04147      0.9942     -0.0605      -0.09361
#> EVZCLS       0.00305      0.0281      0.9206      -0.00748
#> VXFXICLS     0.01021      0.0569      0.0440       0.91713
#> 
#> 
#> LSE for week:
#>           GVZCLS_week  OVXCLS_week  EVZCLS_week  VXFXICLS_week
#> GVZCLS        0.01622      -0.0554      -0.1608         0.0637
#> OVXCLS       -0.07093      -0.0373       0.2000         0.1034
#> EVZCLS       -0.00334      -0.0414      -0.0101         0.0239
#> VXFXICLS     -0.03756      -0.0787      -0.0135         0.0480
#> 
#> 
#> LSE for month:
#>           GVZCLS_month  OVXCLS_month  EVZCLS_month  VXFXICLS_month
#> GVZCLS        0.084981       0.00359        0.0228         -0.0299
#> OVXCLS        0.045986       0.03825       -0.1564         -0.0157
#> EVZCLS       -0.000597       0.02030        0.0501         -0.0138
#> VXFXICLS      0.041648       0.01263        0.0639         -0.0371
#> 
#> 
#> LSE for constant:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    0.491     0.135     0.105     0.926  
#> 
#> 
#> --------------------------------------------------
#> *_day, *_week, *_month of the Coefficient matrix: daily, weekly, and monthly term in the VHAR model
```

``` r
# class----------------
class(fit_har)
#> [1] "vharlse"  "olsmod"   "bvharmod"
# inheritance----------
is.varlse(fit_har)
#> [1] FALSE
is.vharlse(fit_har)
#> [1] TRUE
# complements----------
names(fit_har)
#>  [1] "coefficients"  "fitted.values" "residuals"     "covmat"       
#>  [5] "df"            "m"             "obs"           "y0"           
#>  [9] "p"             "week"          "month"         "totobs"       
#> [13] "process"       "type"          "HARtrans"      "design"       
#> [17] "y"             "method"        "call"
```

### BVAR

This page provides deprecated two functions examples. Both
[`bvar_minnesota()`](../reference/bvar_minnesota.md) and
[`bvar_flat()`](../reference/bvar_flat.md) will be integrated into
[`var_bayes()`](../reference/var_bayes.md) and removed in the next
version.

#### Minnesota prior

- Litterman (1986) and Bańbura et al. (2010)
- All the equations are centered around the random walk with drift.
- *Prior mean*: Recent lags provide more reliable information than the
  more distant ones.
- *Prior variance*: Own lags explain more of the variation of a given
  variable than the lags of other variables in the equation.

First specify the prior using
`set_bvar(sigma, lambda, delta, eps = 1e-04)`.

``` r
bvar_lag <- 5
sig <- apply(etf_train, 2, sd) # sigma vector
lam <- .2 # lambda
delta <- rep(0, m) # delta vector (0 vector since RV stationary)
eps <- 1e-04 # very small number
(bvar_spec <- set_bvar(sig, lam, delta, eps))
#> Model Specification for BVAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: Minnesota
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>     3.77     10.63      2.27      3.81  
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'delta':
#> [1]  0  0  0  0
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

In turn, `bvar_minnesota(y, p, bayes_spec, include_mean = TRUE)` fits
BVAR(p).

- `y`: Multivariate time series data. It should be data frame or matrix,
  which means that every column is numeric. Each column indicates
  variable, i.e. it sould be wide format.
- `p`: Order of BVAR
- `bayes_spec`: Output of [`set_bvar()`](../reference/set_bvar.md)
- `include_mean = TRUE`: By default, you include the constant term in
  the model.

``` r
(fit_bvar <- bvar_minnesota(etf_train, bvar_lag, num_iter = 10, bayes_spec = bvar_spec))
#> Call:
#> bvar_minnesota(y = etf_train, p = bvar_lag, num_iter = 10, bayes_spec = bvar_spec)
#> 
#> BVAR(5) with Minnesota Prior
#> ====================================================
#> 
#> A ~ Matrix Normal (Mean, Precision, Scale = Sigma)
#> ====================================================
#> Matrix Normal Mean for A1 part:
#>           GVZCLS_1  OVXCLS_1  EVZCLS_1  VXFXICLS_1
#> GVZCLS     0.85159   0.00805    0.0380     0.00936
#> OVXCLS     0.04043   0.72094    0.0938     0.01265
#> EVZCLS     0.00441   0.00924    0.8177     0.01768
#> VXFXICLS   0.02356   0.00884    0.1032     0.74819
#> 
#> 
#> Matrix Normal Mean for A2 part:
#>            GVZCLS_2   OVXCLS_2  EVZCLS_2  VXFXICLS_2
#> GVZCLS     4.01e-02  -0.006367  -0.00583     0.00182
#> OVXCLS    -4.53e-05   0.137556   0.02019    -0.03619
#> EVZCLS    -2.48e-03   0.000505   0.05880    -0.00784
#> VXFXICLS   1.08e-02  -0.004247   0.00815     0.11008
#> 
#> 
#> Matrix Normal Mean for A3 part:
#>           GVZCLS_3   OVXCLS_3  EVZCLS_3  VXFXICLS_3
#> GVZCLS     0.02302  -0.003351  -0.00818   -3.14e-03
#> OVXCLS    -0.01472   0.051841   0.00251   -1.53e-02
#> EVZCLS    -0.00225   0.000176   0.02825    5.51e-05
#> VXFXICLS  -0.00767  -0.007094  -0.00120    2.66e-02
#> 
#> 
#> Matrix Normal Mean for A4 part:
#>           GVZCLS_4  OVXCLS_4   EVZCLS_4  VXFXICLS_4
#> GVZCLS     0.01747  -0.00157  -0.006397     0.00301
#> OVXCLS    -0.00843   0.02976   0.003176     0.00680
#> EVZCLS    -0.00200  -0.00029   0.015497     0.00187
#> VXFXICLS  -0.00776  -0.00511   0.000981     0.00909
#> 
#> 
#> Matrix Normal Mean for A5 part:
#>           GVZCLS_5   OVXCLS_5  EVZCLS_5  VXFXICLS_5
#> GVZCLS     0.01422  -1.29e-03  -0.00549     0.00140
#> OVXCLS    -0.00321   2.06e-02   0.00264     0.00820
#> EVZCLS    -0.00184   2.23e-06   0.00989     0.00146
#> VXFXICLS  -0.00391  -2.06e-03   0.00233     0.01210
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    0.686     0.346     0.107     1.300  
#> 
#> 
#> dim(Matrix Normal precision matrix):
#> [1]  21  21
#> 
#> 
#> Sigma ~ Inverse-Wishart
#> ====================================================
#> IW scale matrix:
#>           GVZCLS  OVXCLS  EVZCLS  VXFXICLS
#> GVZCLS      1103     368     111       285
#> OVXCLS       368    3369     125       398
#> EVZCLS       111     125     164       123
#> VXFXICLS     285     398     123      1245
#> 
#> IW degrees of freedom:
#> [1] 887
#> 
#> 
#> --------------------------------------------------
#> *_j of the Coefficient matrix: corresponding to the j-th BVAR lag
```

It is `bvarmn` class. For Bayes computation, it also has other class
such as `normaliw` and `bvharmod`.

``` r
# class---------------
class(fit_bvar)
#> [1] "bvarmn"   "bvharmod" "normaliw"
# inheritance---------
is.bvarmn(fit_bvar)
#> [1] TRUE
# names---------------
names(fit_bvar)
#>  [1] "coefficients"    "fitted.values"   "residuals"       "mn_prec"        
#>  [5] "covmat"          "iw_shape"        "df"              "m"              
#>  [9] "obs"             "prior_mean"      "prior_precision" "prior_scale"    
#> [13] "prior_shape"     "y0"              "design"          "p"              
#> [17] "totobs"          "type"            "y"               "spec"           
#> [21] "chain"           "iter"            "burn"            "thin"           
#> [25] "call"            "process"
```

#### Flat prior

Ghosh et al. (2018) provides flat prior for covariance matrix,
i.e. non-informative. Use `set_bvar_flat(U)`.

``` r
(flat_spec <- set_bvar_flat(U = 5000 * diag(m * bvar_lag + 1))) # c * I
#> Model Specification for BVAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: Flat
#> ========================================================
#> 
#> Setting for 'U':
#> # A matrix:  21 x 21 
#>        [,1]  [,2]  [,3]  [,4]  [,5]
#>  [1,]  5000     0     0     0     0
#>  [2,]     0  5000     0     0     0
#>  [3,]     0     0  5000     0     0
#>  [4,]     0     0     0  5000     0
#>  [5,]     0     0     0     0  5000
#>  [6,]     0     0     0     0     0
#>  [7,]     0     0     0     0     0
#>  [8,]     0     0     0     0     0
#>  [9,]     0     0     0     0     0
#> [10,]     0     0     0     0     0
#> # ... with 11 more rows
```

Then `bvar_flat(y, p, bayes_spec, include_mean = TRUE)`:

``` r
(fit_ghosh <- bvar_flat(etf_train, bvar_lag, num_iter = 10, bayes_spec = flat_spec))
#> Call:
#> bvar_flat(y = etf_train, p = bvar_lag, num_iter = 10, bayes_spec = flat_spec)
#> 
#> BVAR(5) with Flat Prior
#> ====================================================
#> 
#> A ~ Matrix Normal (Mean, U^{-1}, Scale 2 = Sigma)
#> ====================================================
#> Matrix Normal Mean for A1 part:
#>           GVZCLS_1  OVXCLS_1  EVZCLS_1  VXFXICLS_1
#> GVZCLS      0.3128    0.0460    0.0205      0.0457
#> OVXCLS      0.0440    0.4128    0.0186      0.0314
#> EVZCLS      0.0174    0.0230    0.1334      0.0410
#> VXFXICLS    0.0477    0.0464    0.0481      0.3197
#> 
#> 
#> Matrix Normal Mean for A2 part:
#>           GVZCLS_2  OVXCLS_2  EVZCLS_2  VXFXICLS_2
#> GVZCLS     0.19811   0.00287   0.00771      0.0194
#> OVXCLS     0.01595   0.23709   0.00978     -0.0111
#> EVZCLS     0.00425   0.01205   0.11715      0.0201
#> VXFXICLS   0.02725   0.01516   0.03513      0.2162
#> 
#> 
#> Matrix Normal Mean for A3 part:
#>           GVZCLS_3  OVXCLS_3  EVZCLS_3  VXFXICLS_3
#> GVZCLS     0.13884   -0.0153  -0.00102     0.00591
#> OVXCLS    -0.00741    0.1304   0.00361    -0.02452
#> EVZCLS    -0.00353    0.0079   0.10724     0.01150
#> VXFXICLS   0.00797   -0.0146   0.02633     0.14628
#> 
#> 
#> Matrix Normal Mean for A4 part:
#>           GVZCLS_4  OVXCLS_4  EVZCLS_4  VXFXICLS_4
#> GVZCLS     0.11392  -0.01775  -0.00653     0.00884
#> OVXCLS    -0.01626   0.09421   0.00195    -0.00572
#> EVZCLS    -0.00847   0.00565   0.10046     0.01098
#> VXFXICLS  -0.00356  -0.03067   0.02292     0.10552
#> 
#> 
#> Matrix Normal Mean for A5 part:
#>           GVZCLS_5  OVXCLS_5  EVZCLS_5  VXFXICLS_5
#> GVZCLS     0.11282   -0.0208  -0.01028     0.01136
#> OVXCLS    -0.01507    0.1004   0.00155     0.00973
#> EVZCLS    -0.01252    0.0104   0.09667     0.01361
#> VXFXICLS  -0.00492   -0.0215   0.02342     0.10353
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#> 5.49e-03  7.82e-04  8.83e-05  9.82e-03  
#> 
#> 
#> dim(Matrix Normal precision matrix):
#> [1]  21  21
#> 
#> 
#> Sigma ~ Inverse-Wishart
#> ====================================================
#> IW scale matrix:
#>           GVZCLS  OVXCLS  EVZCLS  VXFXICLS
#> GVZCLS      2483     595     209       535
#> OVXCLS       595    3580     216       578
#> EVZCLS       209     216     771       354
#> VXFXICLS     535     578     354      2472
#> 
#> 
#> --------------------------------------------------
#> *_j of the Coefficient matrix: corresponding to the j-th BVAR lag
```

``` r
# class---------------
class(fit_ghosh)
#> [1] "bvarflat" "normaliw" "bvharmod"
# inheritance---------
is.bvarflat(fit_ghosh)
#> [1] TRUE
# names---------------
names(fit_ghosh)
#>  [1] "coefficients"    "fitted.values"   "residuals"       "mn_prec"        
#>  [5] "covmat"          "iw_shape"        "df"              "m"              
#>  [9] "obs"             "prior_mean"      "prior_precision" "y0"             
#> [13] "design"          "y"               "p"               "type"           
#> [17] "chain"           "iter"            "burn"            "thin"           
#> [21] "call"            "process"         "spec"
```

### BVHAR

Consider the VAR(22) form of VHAR.

$$\begin{aligned}
{\mathbf{Y}_{t} = \mathbf{c}} & {+ \left( \Phi^{(d)} + \frac{1}{5}\Phi^{(w)} + \frac{1}{22}\Phi^{(m)} \right)\mathbf{Y}_{t - 1}} \\
 & {+ \left( \frac{1}{5}\Phi^{(w)} + \frac{1}{22}\Phi^{(m)} \right)\mathbf{Y}_{t - 2} + \cdots\left( \frac{1}{5}\Phi^{(w)} + \frac{1}{22}\Phi^{(m)} \right)\mathbf{Y}_{t - 5}} \\
 & {+ \frac{1}{22}\Phi^{(m)}\mathbf{Y}_{t - 6} + \cdots + \frac{1}{22}\Phi^{(m)}\mathbf{Y}_{t - 22}}
\end{aligned}$$

What does Minnesota prior mean in VHAR model?

- All the equations are centered around
  $\mathbf{Y}_{t} + \mathbf{c} + \Phi^{(d)}\mathbf{Y}_{t - 1} + {\mathbf{ϵ}}_{t}$
- RW form: shrink diagonal elements of $\Phi^{(d)}$ toward one
  - $\Phi^{(w)}$ and $\Phi^{(m)}$ to zero
- WN form: $\delta_{i} = 0$

For more simplicity, write coefficient matrices by
$\Phi^{(1)},\Phi^{(2)},\Phi^{(3)}$. If we apply the prior in the same
way, Minnesota moment becomes

$$E\left\lbrack \left( \Phi^{(l)} \right)_{ij} \right\rbrack = \begin{cases}
\delta_{i} & {j = i,\; l = 1} \\
0 & {o/w}
\end{cases}\quad{Var}\left\lbrack \left( \Phi^{(l)} \right)_{ij} \right\rbrack = \begin{cases}
\frac{\lambda^{2}}{l^{2}} & {j = i} \\
{\nu\frac{\lambda^{2}}{l^{2}}\frac{\sigma_{i}^{2}}{\sigma_{j}^{2}}} & {o/w}
\end{cases}$$

We call this VAR-type Minnesota prior or BVHAR-S.

#### BVHAR-S

`set_bvhar(sigma, lambda, delta, eps = 1e-04)` specifies VAR-type
Minnesota prior.

``` r
(bvhar_spec_v1 <- set_bvhar(sig, lam, delta, eps))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VAR
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>     3.77     10.63      2.27      3.81  
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'delta':
#> [1]  0  0  0  0
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

`bvhar_minnesota(y, har = c(5, 22), bayes_spec, include_mean = TRUE)`
can fit BVHAR with this prior. This is the default prior setting.
Similar to above functions, this function will be also integrated into
[`vhar_bayes()`](../reference/vhar_bayes.md) and removed in the next
version.

``` r
(fit_bvhar_v1 <- bvhar_minnesota(etf_train, num_iter = 10, bayes_spec = bvhar_spec_v1))
#> Call:
#> bvhar_minnesota(y = etf_train, num_iter = 10, bayes_spec = bvhar_spec_v1)
#> 
#> BVHAR with Minnesota Prior
#> ====================================================
#> 
#> Phi ~ Matrix Normal (Mean, Scale 1, Scale 2 = Sigma)
#> ====================================================
#> Matrix Normal Mean for day:
#>           GVZCLS_day  OVXCLS_day  EVZCLS_day  VXFXICLS_day
#> GVZCLS        0.7821     0.00591      0.0593        0.0177
#> OVXCLS        0.0414     0.76234      0.1142       -0.0122
#> EVZCLS        0.0109     0.00960      0.7269        0.0212
#> VXFXICLS      0.0248     0.00453      0.0999        0.8055
#> 
#> 
#> Matrix Normal Mean for week:
#>           GVZCLS_week  OVXCLS_week  EVZCLS_week  VXFXICLS_week
#> GVZCLS         0.1150    -0.008383     -0.03183      -0.001635
#> OVXCLS        -0.0208     0.149920      0.02390      -0.000124
#> EVZCLS        -0.0102    -0.000713      0.13070      -0.001506
#> VXFXICLS      -0.0176    -0.011343      0.00258       0.101744
#> 
#> 
#> Matrix Normal Mean for month:
#>           GVZCLS_month  OVXCLS_month  EVZCLS_month  VXFXICLS_month
#> GVZCLS         0.05252      -0.00372       -0.0107        -0.00296
#> OVXCLS         0.00484       0.05009       -0.0239        -0.00967
#> EVZCLS        -0.00439       0.00444        0.0497        -0.00138
#> VXFXICLS       0.00743      -0.00340        0.0112         0.00565
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>   0.6114    0.1271    0.0731    1.1532  
#> 
#> 
#> dim(Matrix Normal precision matrix):
#> [1]  13  13
#> 
#> 
#> Sigma ~ Inverse-Wishart
#> ====================================================
#> IW scale matrix:
#>           GVZCLS  OVXCLS  EVZCLS  VXFXICLS
#> GVZCLS      1263     367     114       286
#> OVXCLS       367    3469     128       379
#> EVZCLS       114     128     218       121
#> VXFXICLS     286     379     121      1183
```

This model is `bvharmn` class.

``` r
# class---------------
class(fit_bvhar_v1)
#> [1] "bvharmn"  "bvharmod" "normaliw"
# inheritance---------
is.bvharmn(fit_bvhar_v1)
#> [1] TRUE
# names---------------
names(fit_bvhar_v1)
#>  [1] "coefficients"    "fitted.values"   "residuals"       "mn_prec"        
#>  [5] "covmat"          "iw_shape"        "df"              "m"              
#>  [9] "obs"             "prior_mean"      "prior_precision" "prior_scale"    
#> [13] "prior_shape"     "y0"              "design"          "p"              
#> [17] "week"            "month"           "totobs"          "type"           
#> [21] "HARtrans"        "y"               "spec"            "chain"          
#> [25] "iter"            "burn"            "thin"            "call"           
#> [29] "process"
```

#### BVHAR-L

Set $\delta_{i}$ for weekly and monthly coefficient matrices in above
Minnesota moments:

$$E\left\lbrack \left( \Phi^{(l)} \right)_{ij} \right\rbrack = \begin{cases}
d_{i} & {j = i,\; l = 1} \\
w_{i} & {j = i,\; l = 2} \\
m_{i} & {j = i,\; l = 3}
\end{cases}$$

i.e. instead of one `delta` vector, set three vector

- `daily`
- `weekly`
- `monthly`

This is called VHAR-type Minnesota prior or BVHAR-L.

`set_weight_bvhar(sigma, lambda, eps, daily, weekly, monthly)` defines
BVHAR-L.

``` r
daily <- rep(.1, m)
weekly <- rep(.1, m)
monthly <- rep(.1, m)
(bvhar_spec_v2 <- set_weight_bvhar(sig, lam, eps, daily, weekly, monthly))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VHAR
#> ========================================================
#> 
#> Setting for 'sigma':
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>     3.77     10.63      2.27      3.81  
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'daily':
#> [1]  0.1  0.1  0.1  0.1
#> 
#> Setting for 'weekly':
#> [1]  0.1  0.1  0.1  0.1
#> 
#> Setting for 'monthly':
#> [1]  0.1  0.1  0.1  0.1
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

`bayes_spec` option of
[`bvhar_minnesota()`](../reference/bvhar_minnesota.md) gets this value,
so you can use this prior intuitively.

``` r
fit_bvhar_v2 <- bvhar_minnesota(
  etf_train,
  num_iter = 10,
  bayes_spec = bvhar_spec_v2
)
fit_bvhar_v2
#> Call:
#> bvhar_minnesota(y = etf_train, num_iter = 10, bayes_spec = bvhar_spec_v2)
#> 
#> BVHAR with Minnesota Prior
#> ====================================================
#> 
#> Phi ~ Matrix Normal (Mean, Scale 1, Scale 2 = Sigma)
#> ====================================================
#> Matrix Normal Mean for day:
#>           GVZCLS_day  OVXCLS_day  EVZCLS_day  VXFXICLS_day
#> GVZCLS       0.86155    -0.00326     -0.0136       -0.0249
#> OVXCLS       0.01731     0.14998      0.1650        0.0612
#> EVZCLS       0.00588     0.01128      0.1482        0.0249
#> VXFXICLS     0.01922     0.00704      0.0432        0.1787
#> 
#> 
#> Matrix Normal Mean for week:
#>           GVZCLS_week  OVXCLS_week  EVZCLS_week  VXFXICLS_week
#> GVZCLS        0.43936    -0.000805     -0.00364       -0.00649
#> OVXCLS        0.00438     0.053457      0.04069        0.01498
#> EVZCLS        0.00138     0.002810      0.06255        0.00597
#> VXFXICLS      0.00453     0.001626      0.01049        0.06789
#> 
#> 
#> Matrix Normal Mean for month:
#>           GVZCLS_month  OVXCLS_month  EVZCLS_month  VXFXICLS_month
#> GVZCLS        0.295313     -0.000218      -0.00153        -0.00287
#> OVXCLS        0.002279      0.030549       0.01689         0.00643
#> EVZCLS        0.000559      0.001287       0.03895         0.00234
#> VXFXICLS      0.001731      0.000646       0.00434         0.03911
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    -9.80     17.31      5.34     16.37  
#> 
#> 
#> dim(Matrix Normal precision matrix):
#> [1]  13  13
#> 
#> 
#> Sigma ~ Inverse-Wishart
#> ====================================================
#> IW scale matrix:
#>           GVZCLS  OVXCLS  EVZCLS  VXFXICLS
#> GVZCLS      5258   -2104    -539     -1765
#> OVXCLS     -2104   63917    8398      5519
#> EVZCLS      -539    8398    2331      2235
#> VXFXICLS   -1765    5519    2235      7440
```
