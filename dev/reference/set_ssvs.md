# Stochastic Search Variable Selection (SSVS) Hyperparameter for Coefficients Matrix and Cholesky Factor

Set SSVS hyperparameters for VAR or VHAR coefficient matrix and Cholesky
factor.

## Usage

``` r
set_ssvs(
  spike_grid = 100L,
  slab_shape = 0.01,
  slab_scl = 0.01,
  s1 = c(1, 1),
  s2 = c(1, 1),
  shape = 0.01,
  rate = 0.01
)

# S3 method for class 'ssvsinput'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.ssvsinput(x)

# S3 method for class 'ssvsinput'
knit_print(x, ...)
```

## Arguments

- spike_grid:

  Griddy gibbs grid size for scaling factor (between 0 and 1) of spike
  sd which is Spike sd = c \* slab sd

- slab_shape:

  Inverse gamma shape for slab sd

- slab_scl:

  Inverse gamma scale for slab sd

- s1:

  First shape of coefficients prior beta distribution

- s2:

  Second shape of coefficients prior beta distribution

- shape:

  Gamma shape parameters for precision matrix (See Details).

- rate:

  Gamma rate parameters for precision matrix (See Details).

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

`ssvsinput` object

## Details

Let \\\alpha\\ be the vectorized coefficient, \\\alpha = vec(A)\\.
Spike-slab prior is given using two normal distributions. \$\$\alpha_j
\mid \gamma_j \sim (1 - \gamma_j) N(0, \tau\_{0j}^2) + \gamma_j N(0,
\tau\_{1j}^2)\$\$ As spike-slab prior itself suggests, set
\\\tau\_{0j}\\ small (point mass at zero: spike distribution) and set
\\\tau\_{1j}\\ large (symmetric by zero: slab distribution).

\\\gamma_j\\ is the proportion of the nonzero coefficients and it
follows \$\$\gamma_j \sim Bernoulli(p_j)\$\$

- `coef_spike`: \\\tau\_{0j}\\

- `coef_slab`: \\\tau\_{1j}\\

- `coef_mixture`: \\p_j\\

- \\j = 1, \ldots, mk\\: vectorized format corresponding to coefficient
  matrix

- If one value is provided, model function will read it by replicated
  value.

- `coef_non`: vectorized constant term is given prior Normal
  distribution with variance \\cI\\. Here, `coef_non` is \\\sqrt{c}\\.

Next for precision matrix \\\Sigma_e^{-1}\\, SSVS applies Cholesky
decomposition. \$\$\Sigma_e^{-1} = \Psi \Psi^T\$\$ where \\\Psi =
\\\psi\_{ij}\\\\ is upper triangular.

Diagonal components follow the gamma distribution. \$\$\psi\_{jj}^2 \sim
Gamma(shape = a_j, rate = b_j)\$\$ For each row of off-diagonal
(upper-triangular) components, we apply spike-slab prior again.
\$\$\psi\_{ij} \mid w\_{ij} \sim (1 - w\_{ij}) N(0, \kappa\_{0,ij}^2) +
w\_{ij} N(0, \kappa\_{1,ij}^2)\$\$ \$\$w\_{ij} \sim
Bernoulli(q\_{ij})\$\$

- `shape`: \\a_j\\

- `rate`: \\b_j\\

- `chol_spike`: \\\kappa\_{0,ij}\\

- `chol_slab`: \\\kappa\_{1,ij}\\

- `chol_mixture`: \\q\_{ij}\\

- \\j = 1, \ldots, mk\\: vectorized format corresponding to coefficient
  matrix

- \\i = 1, \ldots, j - 1\\ and \\j = 2, \ldots, m\\: \\\eta =
  (\psi\_{12}, \psi\_{13}, \psi\_{23}, \psi\_{14}, \ldots, \psi\_{34},
  \ldots, \psi\_{1m}, \ldots, \psi\_{m - 1, m})^T\\

- `chol_` arguments can be one value for replication, vector, or upper
  triangular matrix.

## References

George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs
Sampling*. Journal of the American Statistical Association, 88(423),
881-889.

George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for
VAR model restrictions*. Journal of Econometrics, 142(1), 553-580.

Ishwaran, H., & Rao, J. S. (2005). *Spike and slab variable selection:
Frequentist and Bayesian strategies*. The Annals of Statistics, 33(2).

Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series
Methods for Empirical Macroeconomics*. Foundations and Trends® in
Econometrics, 3(4), 267-358.
