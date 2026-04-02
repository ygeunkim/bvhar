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
#> GVZCLS     3.66752   -0.0246  -0.01890     -0.0376
#> OVXCLS     0.00597    0.5117   0.02546      0.0118
#> EVZCLS    -0.01287   -0.0701   3.71469     -0.0279
#> VXFXICLS   0.00768    0.0103   0.00876      0.4313
#> 
#> 
#> Matrix Normal Mean for A2 part:
#>           GVZCLS_2  OVXCLS_2  EVZCLS_2  VXFXICLS_2
#> GVZCLS    -0.02105  -0.00567  -0.00461    -0.00915
#> OVXCLS     0.00151   0.02407   0.00633     0.00287
#> EVZCLS    -0.00315  -0.01758  -0.01005    -0.00686
#> VXFXICLS   0.00190   0.00250   0.00217     0.00405
#> 
#> 
#> Matrix Normal Mean for A3 part:
#>            GVZCLS_3  OVXCLS_3   EVZCLS_3  VXFXICLS_3
#> GVZCLS    -0.008931  -0.00226  -0.001998    -0.00399
#> OVXCLS     0.000660   0.01060   0.002791     0.00126
#> EVZCLS    -0.001371  -0.00783  -0.004385    -0.00297
#> VXFXICLS   0.000813   0.00106   0.000949     0.00172
#> 
#> 
#> Matrix Normal Mean for A4 part:
#>            GVZCLS_4   OVXCLS_4   EVZCLS_4  VXFXICLS_4
#> GVZCLS    -0.004832  -0.001136  -0.001091   -0.002192
#> OVXCLS     0.000374   0.005927   0.001561    0.000724
#> EVZCLS    -0.000758  -0.004414  -0.002432   -0.001643
#> VXFXICLS   0.000440   0.000569   0.000528    0.000925
#> 
#> 
#> Matrix Normal Mean for A5 part:
#>            GVZCLS_5   OVXCLS_5   EVZCLS_5  VXFXICLS_5
#> GVZCLS    -0.002995  -0.000653  -0.000679   -0.001389
#> OVXCLS     0.000243   0.003779   0.000993    0.000467
#> EVZCLS    -0.000474  -0.002824  -0.001535   -0.001039
#> VXFXICLS   0.000275   0.000359   0.000335    0.000578
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    -45.7      11.7     -20.1      13.1  
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
#> GVZCLS     93178   -5962   13507     -7718
#> OVXCLS     -5962   26271  -17704      2940
#> EVZCLS     13507  -17704   28915     -6057
#> VXFXICLS   -7718    2940   -6057      4479
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
#> GVZCLS        0.0758     0.00341      0.0277        0.0319
#> OVXCLS        0.0208     0.09752      0.2413        0.0538
#> EVZCLS        0.0042     0.00692      0.4932        0.0130
#> VXFXICLS      0.0259     0.00785      0.0684        0.0855
#> 
#> 
#> Matrix Normal Mean for week:
#>           GVZCLS_week  OVXCLS_week  EVZCLS_week  VXFXICLS_week
#> GVZCLS       0.014682      0.00067      0.00644        0.00769
#> OVXCLS       0.005318      0.02151      0.05957        0.01319
#> EVZCLS       0.000997      0.00172      0.00929        0.00314
#> VXFXICLS     0.006168      0.00183      0.01669        0.01735
#> 
#> 
#> Matrix Normal Mean for month:
#>           GVZCLS_month  OVXCLS_month  EVZCLS_month  VXFXICLS_month
#> GVZCLS        0.005478      9.34e-05       0.00243         0.00301
#> OVXCLS        0.002801      9.08e-03       0.02477         0.00566
#> EVZCLS        0.000414      7.92e-04       0.00385         0.00124
#> VXFXICLS      0.002377      7.25e-04       0.00694         0.00571
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    15.02     19.54      3.63     20.23  
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
#> GVZCLS     10852    3444     764      4366
#> OVXCLS      3444   80313    6335      7362
#> EVZCLS       764    6335    1086      1815
#> VXFXICLS    4366    7362    1815     10482
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
#> GVZCLS       0.64282    -0.00111    -0.00429      -0.00456
#> OVXCLS       0.01762     0.17285     0.19101       0.04454
#> EVZCLS       0.00644     0.01049     0.15609       0.01968
#> VXFXICLS     0.02128     0.00664     0.05535       0.15807
#> 
#> 
#> Matrix Normal Mean for week:
#>           GVZCLS_week  OVXCLS_week  EVZCLS_week  VXFXICLS_week
#> GVZCLS        0.30409    -0.000323     -0.00141       -0.00135
#> OVXCLS        0.00443     0.067864      0.04711        0.01087
#> EVZCLS        0.00152     0.002611      0.06353        0.00472
#> VXFXICLS      0.00502     0.001533      0.01345        0.06186
#> 
#> 
#> Matrix Normal Mean for month:
#>           GVZCLS_month  OVXCLS_month  EVZCLS_month  VXFXICLS_month
#> GVZCLS        0.203318     -0.000112     -0.000669       -0.000697
#> OVXCLS        0.002299      0.040851      0.019538        0.004635
#> EVZCLS        0.000613      0.001194      0.038892        0.001847
#> VXFXICLS      0.001920      0.000607      0.005568        0.036314
#> 
#> 
#> Matrix Normal Mean for constant part:
#>   GVZCLS    OVXCLS    EVZCLS  VXFXICLS  
#>    -2.51     16.19      5.45     16.91  
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
#> GVZCLS    1428.4    -478   -53.7      -199
#> OVXCLS    -478.3   55491  7830.4      5257
#> EVZCLS     -53.7    7830  2353.9      2300
#> VXFXICLS  -199.4    5257  2300.2      7729
```
