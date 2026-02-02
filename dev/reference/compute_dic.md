# Deviance Information Criterion of Multivariate Time Series Model

Compute DIC of BVAR and BVHAR.

## Usage

``` r
compute_dic(object, ...)

# S3 method for class 'bvarmn'
compute_dic(object, n_iter = 100L, ...)
```

## Arguments

- object:

  Model fit

- ...:

  not used

- n_iter:

  Number to sample

## Value

DIC value.

## Details

Deviance information criteria (DIC) is

\$\$- 2 \log p(y \mid \hat\theta\_{bayes}) + 2 p\_{DIC}\$\$

where \\p\_{DIC}\\ is the effective number of parameters defined by

\$\$p\_{DIC} = 2 ( \log p(y \mid \hat\theta\_{bayes}) - E\_{post} \log
p(y \mid \theta) )\$\$

Random sampling from posterior distribution gives its computation,
\\\theta_i \sim \theta \mid y, i = 1, \ldots, M\\

\$\$p\_{DIC}^{computed} = 2 ( \log p(y \mid \hat\theta\_{bayes}) -
\frac{1}{M} \sum_i \log p(y \mid \theta_i) )\$\$

## References

Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013).
*Bayesian data analysis*. Chapman and Hall/CRC.

Spiegelhalter, D.J., Best, N.G., Carlin, B.P. and Van Der Linde, A.
(2002). *Bayesian measures of model complexity and fit*. Journal of the
Royal Statistical Society: Series B (Statistical Methodology), 64:
583-639.
