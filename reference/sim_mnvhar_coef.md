# Generate Minnesota BVAR Parameters

This function generates parameters of BVAR with Minnesota prior.

## Usage

``` r
sim_mnvhar_coef(bayes_spec = set_bvhar(), full = TRUE)
```

## Arguments

- bayes_spec:

  A BVHAR model specification by [`set_bvhar()`](set_bvar.md) (default)
  or [`set_weight_bvhar()`](set_bvar.md).

- full:

  Generate variance matrix from IW (default: `TRUE`) or not (`FALSE`)?

## Value

List with the following component.

- coefficients:

  BVHAR coefficient (MN)

- covmat:

  BVHAR variance (IW or diagonal matrix of `sigma` of `bayes_spec`)

## Details

Normal-IW family for vector HAR model: \$\$\Phi \mid \Sigma_e \sim
MN(M_0, \Omega_0, \Sigma_e)\$\$ \$\$\Sigma_e \sim IW(\Psi_0, \nu_0)\$\$

## References

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

## See also

- [`set_bvhar()`](set_bvar.md) to specify the hyperparameters of
  VAR-type Minnesota prior.

- [`set_weight_bvhar()`](set_bvar.md) to specify the hyperparameters of
  HAR-type Minnesota prior.

## Examples

``` r
# Generate (Phi, Sigma)
# BVHAR-S
# sigma: 1, 1, 1
# lambda: .1
# delta: .1, .1, .1
# epsilon: 1e-04
set.seed(1)
sim_mnvhar_coef(
  bayes_spec = set_bvhar(
    sigma = rep(1, 3),
    lambda = .1,
    delta = rep(.1, 3),
    eps = 1e-04
  ),
  full = TRUE
)
#> $coefficients
#>                [,1]          [,2]         [,3]
#>  [1,]  0.1760184710 -0.0165328928 -0.065147252
#>  [2,]  0.1556541344  0.0361635386 -0.167462417
#>  [3,] -0.2280271782  0.1579619956  0.348108349
#>  [4,] -0.0008334809  0.0183525644  0.051859820
#>  [5,]  0.0305742650  0.0020105043  0.021978150
#>  [6,]  0.0038386336 -0.0397341846 -0.023663045
#>  [7,] -0.0019263525 -0.0009998963 -0.037044795
#>  [8,] -0.0164102306  0.0135852939  0.054350260
#>  [9,] -0.0035277008  0.0066925905  0.007425529
#> 
#> $covmat
#>            [,1]       [,2]       [,3]
#> [1,]  1.0600918 -0.5356457 -0.9412103
#> [2,] -0.5356457  0.4150287  0.6410763
#> [3,] -0.9412103  0.6410763  1.5794713
#> 
```
