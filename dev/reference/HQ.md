# Hannan-Quinn Criterion

Compute HQ of VAR(p), VHAR, BVAR(p), and BVHAR

## Usage

``` r
HQ(object, ...)

# S3 method for class 'logLik'
HQ(object, ...)

# S3 method for class 'varlse'
HQ(object, ...)

# S3 method for class 'vharlse'
HQ(object, ...)

# S3 method for class 'bvarmn'
HQ(object, ...)

# S3 method for class 'bvarflat'
HQ(object, ...)

# S3 method for class 'bvharmn'
HQ(object, ...)
```

## Arguments

- object:

  A `logLik` object or Model fit

- ...:

  not used

## Value

HQ value.

## Details

The formula is

\$\$HQ = -2 \log p(y \mid \hat\theta) + k \log\log(T)\$\$

which can be computed by
`AIC(object, ..., k = 2 * log(log(nobs(object))))` with
[`stats::AIC()`](https://rdrr.io/r/stats/AIC.html).

Let \\\tilde{\Sigma}\_e\\ be the MLE and let \\\hat{\Sigma}\_e\\ be the
unbiased estimator (`covmat`) for \\\Sigma_e\\. Note that

\$\$\tilde{\Sigma}\_e = \frac{n - k}{T} \hat{\Sigma}\_e\$\$

Then

\$\$HQ(p) = \log \det \Sigma_e + \frac{2 \log \log n}{n}(\text{number of
freely estimated parameters})\$\$

where the number of freely estimated parameters is \\pm^2\\.

## References

Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of
an Autoregression*. Journal of the Royal Statistical Society: Series B
(Methodological), 41: 190-195.

Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of
an Autoregression*. Journal of the Royal Statistical Society: Series B
(Methodological), 41: 190-195.

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

Quinn, B.G. (1980). *Order Determination for a Multivariate
Autoregression*. Journal of the Royal Statistical Society: Series B
(Methodological), 42: 182-185.
