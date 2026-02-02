# Extracting Log of Marginal Likelihood

Compute log of marginal likelihood of Bayesian Fit

## Usage

``` r
compute_logml(object, ...)

# S3 method for class 'bvarmn'
compute_logml(object, ...)

# S3 method for class 'bvharmn'
compute_logml(object, ...)
```

## Arguments

- object:

  Model fit

- ...:

  not used

## Value

log likelihood of Minnesota prior model.

## Details

Closed form of Marginal Likelihood of BVAR can be derived by

\$\$p(Y_0) = \pi^{-mn / 2} \frac{\Gamma_m ((\alpha_0 + n) / 2)}{\Gamma_m
(\alpha_0 / 2)} \det(\Omega_0)^{-m / 2} \det(S_0)^{\alpha_0 / 2}
\det(\hat{V})^{- m / 2} \det(\hat{\Sigma}\_e)^{-(\alpha_0 + n) / 2}\$\$

Closed form of Marginal Likelihood of BVHAR can be derived by

\$\$p(Y_0) = \pi^{-ms_0 / 2} \frac{\Gamma_m ((d_0 + n) / 2)}{\Gamma_m
(d_0 / 2)} \det(P_0)^{-m / 2} \det(U_0)^{d_0 / 2}
\det(\hat{V}\_{HAR})^{- m / 2} \det(\hat{\Sigma}\_e)^{-(d_0 + n) /
2}\$\$

## References

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for
Vector Autoregressions*. Review of Economics and Statistics, 97(2).
