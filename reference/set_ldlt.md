# Covariance Matrix Prior Specification

**\[experimental\]** Set prior for covariance matrix.

## Usage

``` r
set_ldlt(ig_shape = 3, ig_scl = 0.01)

set_sv(ig_shape = 3, ig_scl = 0.01, initial_mean = 1, initial_prec = 0.1)

# S3 method for class 'covspec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.covspec(x)

is.svspec(x)

is.ldltspec(x)
```

## Arguments

- ig_shape:

  Inverse-Gamma shape of Cholesky diagonal vector. For SV (`set_sv()`),
  this is for state variance.

- ig_scl:

  Inverse-Gamma scale of Cholesky diagonal vector. For SV (`set_sv()`),
  this is for state variance.

- initial_mean:

  Prior mean of initial state.

- initial_prec:

  Prior precision of initial state.

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Details

`set_ldlt()` specifies LDLT of precision matrix, \$\$\Sigma^{-1} = L^T
D^{-1} L\$\$

`set_sv()` specifices time varying precision matrix under stochastic
volatility framework based on \$\$\Sigma_t^{-1} = L^T D_t^{-1} L\$\$

## References

Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022).
*Corrigendum to “Large Bayesian vector autoregressions with stochastic
volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019)
137-154\]*. Journal of Econometrics, 227(2), 506-512.

Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian
Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge:
Cambridge University Press.
