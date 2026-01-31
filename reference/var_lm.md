# Fitting Vector Autoregressive Model of Order p Model

This function fits VAR(p) using OLS method.

## Usage

``` r
var_lm(
  y,
  p = 1,
  exogen = NULL,
  s = 0,
  include_mean = TRUE,
  method = c("nor", "chol", "qr")
)

# S3 method for class 'varlse'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'varlse'
logLik(object, ...)

# S3 method for class 'varlse'
AIC(object, ...)

# S3 method for class 'varlse'
BIC(object, ...)

is.varlse(x)

is.bvharmod(x)

# S3 method for class 'varlse'
knit_print(x, ...)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- p:

  Lag of VAR (Default: 1)

- exogen:

  Exogenous variables

- s:

  Lag of exogeneous variables in VARX(p, s). By default, `s = 0`.

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- method:

  Method to solve linear equation system. (`nor`: normal equation
  (default), `chol`: Cholesky, and `qr`: HouseholderQR)

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

- object:

  A `varlse` object

## Value

`var_lm()` returns an object named `varlse`
[class](https://rdrr.io/r/base/class.html). It is a list with the
following components:

- coefficients:

  Coefficient Matrix

- fitted.values:

  Fitted response values

- residuals:

  Residuals

- covmat:

  LS estimate for covariance matrix

- df:

  Numer of Coefficients

- p:

  Lag of VAR

- m:

  Dimension of the data

- obs:

  Sample size used when training = `totobs` - `p`

- totobs:

  Total number of the observation

- call:

  Matched call

- process:

  Process: VAR

- type:

  include constant term (`const`) or not (`none`)

- design:

  Design matrix

- y:

  Raw input

- y0:

  Multivariate response matrix

- method:

  Solving method

- call:

  Matched call

It is also a `bvharmod` class.

## Details

This package specifies VAR(p) model as

\$\$Y\_{t} = A_1 Y\_{t - 1} + \cdots + A_p Y\_{t - p} + c +
\epsilon_t\$\$

If `include_type = TRUE`, there is constant term. The function estimates
every coefficient matrix.

Consider the response matrix \\Y_0\\. Let \\T\\ be the total number of
sample, let \\m\\ be the dimension of the time series, let \\p\\ be the
order of the model, and let \\n = T - p\\. Likelihood of VAR(p) has

\$\$Y_0 \mid B, \Sigma_e \sim MN(X_0 B, I_s, \Sigma_e)\$\$

where \\X_0\\ is the design matrix, and MN is [matrix normal
distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution).

Then log-likelihood of vector autoregressive model family is specified
by

\$\$\log p(Y_0 \mid B, \Sigma_e) = - \frac{nm}{2} \log 2\pi -
\frac{n}{2} \log \det \Sigma_e - \frac{1}{2} tr( (Y_0 - X_0 B)
\Sigma_e^{-1} (Y_0 - X_0 B)^T )\$\$

In addition, recall that the OLS estimator for the matrix coefficient
matrix is the same as MLE under the Gaussian assumption. MLE for
\\\Sigma_e\\ has different denominator, \\n\\.

\$\$\hat{B} = \hat{B}^{LS} = \hat{B}^{ML} = (X_0^T X_0)^{-1} X_0^T
Y_0\$\$ \$\$\hat\Sigma_e = \frac{1}{s - k} (Y_0 - X_0 \hat{B})^T (Y_0 -
X_0 \hat{B})\$\$ \$\$\tilde\Sigma_e = \frac{1}{s} (Y_0 - X_0 \hat{B})^T
(Y_0 - X_0 \hat{B}) = \frac{s - k}{s} \hat\Sigma_e\$\$

Let \\\tilde{\Sigma}\_e\\ be the MLE and let \\\hat{\Sigma}\_e\\ be the
unbiased estimator (`covmat`) for \\\Sigma_e\\. Note that

\$\$\tilde{\Sigma}\_e = \frac{n - k}{n} \hat{\Sigma}\_e\$\$

Then

\$\$AIC(p) = \log \det \Sigma_e + \frac{2}{n}(\text{number of freely
estimated parameters})\$\$

where the number of freely estimated parameters is \\mk\\, i.e. \\pm^2\\
or \\pm^2 + m\\.

Let \\\tilde{\Sigma}\_e\\ be the MLE and let \\\hat{\Sigma}\_e\\ be the
unbiased estimator (`covmat`) for \\\Sigma_e\\. Note that

\$\$\tilde{\Sigma}\_e = \frac{n - k}{T} \hat{\Sigma}\_e\$\$

Then

\$\$BIC(p) = \log \det \Sigma_e + \frac{\log n}{n}(\text{number of
freely estimated parameters})\$\$

where the number of freely estimated parameters is \\pm^2\\.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

Akaike, H. (1969). *Fitting autoregressive models for prediction*. Ann
Inst Stat Math 21, 243-247.

Akaike, H. (1971). *Autoregressive model fitting for control*. Ann Inst
Stat Math 23, 163-180.

Akaike H. (1974). *A new look at the statistical model identification*.
IEEE Transactions on Automatic Control, vol. 19, no. 6, pp. 716-723.

Akaike H. (1998). *Information Theory and an Extension of the Maximum
Likelihood Principle*. In: Parzen E., Tanabe K., Kitagawa G. (eds)
Selected Papers of Hirotugu Akaike. Springer Series in Statistics
(Perspectives in Statistics). Springer, New York, NY.

Gideon Schwarz. (1978). *Estimating the Dimension of a Model*. Ann.
Statist. 6 (2) 461 - 464.

## See also

- [`summary.varlse()`](summary.varlse.md) to summarize VAR model

## Examples

``` r
# Perform the function using etf_vix dataset
fit <- var_lm(y = etf_vix, p = 2)
class(fit)
#> [1] "varlse"   "olsmod"   "bvharmod"
str(fit)
#> List of 16
#>  $ coefficients : num [1:19, 1:9] 0.9588 0.0313 -0.0235 -0.0944 -0.0048 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:19] "GVZCLS_1" "OVXCLS_1" "VXFXICLS_1" "VXEEMCLS_1" ...
#>   .. ..$ : chr [1:9] "GVZCLS" "OVXCLS" "VXFXICLS" "VXEEMCLS" ...
#>  $ fitted.values: num [1:903, 1:9] 21.3 22.2 21.5 21.1 21.3 ...
#>  $ residuals    : num [1:903, 1:9] 1.008 -0.639 -0.284 0.298 0.308 ...
#>  $ covmat       : num [1:9, 1:9] 1.113 0.373 0.304 0.445 1.366 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:9] "GVZCLS" "OVXCLS" "VXFXICLS" "VXEEMCLS" ...
#>   .. ..$ : chr [1:9] "GVZCLS" "OVXCLS" "VXFXICLS" "VXEEMCLS" ...
#>  $ df           : int 19
#>  $ m            : int 9
#>  $ obs          : int 903
#>  $ y0           : num [1:903, 1:9] 22.3 21.6 21.2 21.4 21.6 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr [1:9] "GVZCLS" "OVXCLS" "VXFXICLS" "VXEEMCLS" ...
#>  $ p            : int 2
#>  $ totobs       : num 905
#>  $ process      : chr "VAR"
#>  $ type         : chr "const"
#>  $ design       : num [1:903, 1:19] 21.5 22.3 21.6 21.2 21.4 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr [1:19] "GVZCLS_1" "OVXCLS_1" "VXFXICLS_1" "VXEEMCLS_1" ...
#>  $ y            : num [1:905, 1:9] 21.5 21.5 22.3 21.6 21.2 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr [1:9] "GVZCLS" "OVXCLS" "VXFXICLS" "VXEEMCLS" ...
#>  $ method       : chr "nor"
#>  $ call         : language var_lm(y = etf_vix, p = 2)
#>  - attr(*, "class")= chr [1:3] "varlse" "olsmod" "bvharmod"

# Extract coef, fitted values, and residuals
coef(fit)
#>                  GVZCLS       OVXCLS     VXFXICLS    VXEEMCLS     VXSLVCLS
#> GVZCLS_1    0.958839561 -0.062827385 -0.019710690  0.01257027  0.291868669
#> OVXCLS_1    0.031306519  0.946200622 -0.006632872 -0.03089072  0.005701801
#> VXFXICLS_1 -0.023546344 -0.052053167  0.853672620 -0.06031077 -0.096749298
#> VXEEMCLS_1 -0.094354618 -0.073106564  0.074752695  0.86705288 -0.048927384
#> VXSLVCLS_1 -0.004803889  0.011100124 -0.028180528 -0.00787754  0.741225547
#> EVZCLS_1    0.125516130 -0.138276981  0.077667943 -0.05620528  0.213417413
#> VXXLECLS_1  0.077038206  0.116602536  0.049019313  0.19068844  0.123664057
#> VXGDXCLS_1 -0.041364201  0.022893379 -0.012790918 -0.02444514 -0.051351386
#> VXEWZCLS_1  0.032078312  0.081017743  0.002689773  0.02421098  0.011793646
#> GVZCLS_2   -0.104834812 -0.017750378 -0.063640299 -0.06563731 -0.284901251
#> OVXCLS_2   -0.057264753  0.000198619 -0.012230807  0.01529842 -0.021755404
#> VXFXICLS_2 -0.010770464  0.007308545  0.070962129  0.06532811  0.035457057
#> VXEEMCLS_2  0.109941832  0.031515218 -0.087805109  0.05027743  0.106490192
#> VXSLVCLS_2  0.031836190  0.026261295  0.084884303  0.07330380  0.165164782
#> EVZCLS_2   -0.039927065  0.240594601  0.018595199  0.07171981 -0.095041473
#> VXXLECLS_2 -0.063856767 -0.045562528 -0.052978615 -0.18149299 -0.106494854
#> VXGDXCLS_2  0.084605713  0.007572189  0.029414253  0.02242848  0.085005069
#> VXEWZCLS_2 -0.033916295 -0.071267512  0.007601670 -0.02085634 -0.019106806
#> const       0.484435224  0.071858917  0.764593553  0.73643928  0.993267584
#>                   EVZCLS     VXXLECLS     VXGDXCLS     VXEWZCLS
#> GVZCLS_1   -0.0070051856  0.010258469  0.186062079  0.016397003
#> OVXCLS_1    0.0057743474 -0.055730279 -0.008620592 -0.017425642
#> VXFXICLS_1 -0.0013000787 -0.071102522 -0.019267681 -0.093669606
#> VXEEMCLS_1  0.0082641013  0.043573682  0.052515700  0.111449434
#> VXSLVCLS_1  0.0048970514 -0.010365439 -0.037485186 -0.035578884
#> EVZCLS_1    0.9530755788 -0.159179164 -0.029104883 -0.109896535
#> VXXLECLS_1  0.0073924911  1.090977549  0.173739155  0.097903592
#> VXGDXCLS_1 -0.0170045623 -0.002534079  0.717661085  0.014081721
#> VXEWZCLS_1  0.0295958720  0.039727624  0.029208637  0.919861136
#> GVZCLS_2    0.0011140822 -0.064661782 -0.129207944 -0.025690608
#> OVXCLS_2   -0.0013553534  0.045540919  0.007661278  0.007209753
#> VXFXICLS_2  0.0193175978  0.067148832  0.057541033  0.120471274
#> VXEEMCLS_2 -0.0273314378 -0.088654468 -0.128280935 -0.177756749
#> VXSLVCLS_2  0.0074048747  0.057079506  0.076151265  0.072552465
#> EVZCLS_2   -0.0026576824  0.171823659 -0.062484101  0.077971334
#> VXXLECLS_2 -0.0007146444 -0.109371143 -0.146653392 -0.060460831
#> VXGDXCLS_2  0.0111059419  0.007634580  0.222052920 -0.026452775
#> VXEWZCLS_2 -0.0304841577 -0.030496662 -0.018978557  0.055930225
#> const       0.1341878224  0.742494021  0.705738212  0.796582789
head(residuals(fit))
#>            [,1]        [,2]       [,3]        [,4]       [,5]        [,6]
#> [1,]  1.0084235 -0.01665024 -0.1189584  0.18259423  1.0345739  0.54161189
#> [2,] -0.6393271  1.07781122 -0.7889172  0.06023516 -0.2598844 -0.30429270
#> [3,] -0.2841251 -1.13853760  0.8285284  1.09281272  1.3066535  0.53399638
#> [4,]  0.2975186 -0.89689641 -0.5797484 -0.31164787  1.3474286 -0.09610840
#> [5,]  0.3075188 -0.98987446 -0.6230728 -0.46134610  1.2499047  0.01799667
#> [6,] -0.4052045 -1.62792068 -0.9718923 -2.26508501 -1.1137253 -0.25997035
#>             [,7]       [,8]       [,9]
#> [1,]  0.97639374  0.3811097 -0.2513162
#> [2,]  0.08960441 -0.6696317  0.4110768
#> [3,]  0.24450709 -0.9864375  1.7865058
#> [4,] -0.20575733  0.6892149 -0.3931113
#> [5,] -0.21831258  1.3123282 -0.5570114
#> [6,] -0.18371357 -1.1068942 -1.1856081
head(fitted(fit))
#>          [,1]     [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
#> [1,] 21.33158 35.53665 29.17896 29.63741 42.47543 12.57839 26.67361 33.37889
#> [2,] 22.23933 35.51219 29.24892 29.94976 43.08988 13.07429 27.55040 34.16963
#> [3,] 21.48413 36.75854 28.71147 30.03719 42.17335 12.77600 27.55549 33.79644
#> [4,] 21.10248 35.69190 29.68475 30.97165 42.64257 13.32611 27.66076 33.26579
#> [5,] 21.29248 34.95987 29.29307 30.65135 43.25010 13.13200 27.32831 33.78767
#> [6,] 21.54520 34.21792 28.93189 30.24509 43.75373 13.04997 27.03371 34.90689
#>          [,9]
#> [1,] 30.43132
#> [2,] 30.33892
#> [3,] 30.91349
#> [4,] 32.54811
#> [5,] 32.16701
#> [6,] 31.65561
```
