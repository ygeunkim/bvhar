# Hyperpriors for Bayesian Models

Set hyperpriors of Bayesian VAR and VHAR models.

## Usage

``` r
set_lambda(
  mode = 0.2,
  sd = 0.4,
  param = NULL,
  lower = 1e-05,
  upper = 3,
  grid_size = 100L
)

set_psi(shape = 4e-04, scale = 4e-04, lower = 1e-05, upper = 3)

# S3 method for class 'bvharpriorspec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.bvharpriorspec(x)

# S3 method for class 'bvharpriorspec'
knit_print(x, ...)
```

## Arguments

- mode:

  Mode of Gamma distribution. By default, `.2`.

- sd:

  Standard deviation of Gamma distribution. By default, `.4`.

- param:

  Shape and rate of Gamma distribution, in the form of `c(shape, rate)`.
  If specified, ignore `mode` and `sd`.

- lower:

  **\[experimental\]** Lower bound for
  [`stats::optim()`](https://rdrr.io/r/stats/optim.html). By default,
  `1e-5`.

- upper:

  **\[experimental\]** Upper bound for
  [`stats::optim()`](https://rdrr.io/r/stats/optim.html). By default,
  `3`.

- grid_size:

  Griddy gibbs grid size for lag scaling

- shape:

  Shape of Inverse Gamma distribution. By default, `(.02)^2`.

- scale:

  Scale of Inverse Gamma distribution. By default, `(.02)^2`.

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

`bvharpriorspec` object

## Details

In addition to Normal-IW priors [`set_bvar()`](set_bvar.md),
[`set_bvhar()`](set_bvar.md), and [`set_weight_bvhar()`](set_bvar.md),
these functions give hierarchical structure to the model.

- `set_lambda()` specifies hyperprior for \\\lambda\\ (`lambda`), which
  is Gamma distribution.

- `set_psi()` specifies hyperprior for \\\psi / (\nu_0 - k - 1) =
  \sigma^2\\ (`sigma`), which is Inverse gamma distribution.

The following set of `(mode, sd)` are recommended by Sims and Zha (1998)
for `set_lambda()`.

- `(mode = .2, sd = .4)`: default

- `(mode = 1, sd = 1)`

Giannone et al. (2015) suggested data-based selection for `set_psi()`.
It chooses (0.02)^2 based on its empirical data set.

## References

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for
Vector Autoregressions*. Review of Economics and Statistics, 97(2).

## Examples

``` r
# Hirearchical BVAR specification------------------------
set_bvar(
  sigma = set_psi(shape = 4e-4, scale = 4e-4),
  lambda = set_lambda(mode = .2, sd = .4),
  delta = rep(1, 3),
  eps = 1e-04 # eps = 1e-04
)
#> Model Specification for BVAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_Hierarchical
#> ========================================================
#> 
#> Setting for 'sigma':
#> Hyperprior specification for psi
#> 
#> [1]  psi ~ Inv-Gamma(shape = 4e-04, scale =4e-04)
#> 
#> Setting for 'lambda':
#> Hyperprior specification for lambda
#> 
#> [1]  lambda ~ Gamma(shape = 1.64038820320221, rate =3.20194101601104)
#> 
#> Setting for 'delta':
#> [1]  1  1  1
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'hierarchical':
#> [1]  TRUE
#> 
```
