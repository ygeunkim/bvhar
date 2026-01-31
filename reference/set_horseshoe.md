# Horseshoe Prior Specification

Set initial hyperparameters and parameter before starting Gibbs sampler
for Horseshoe prior.

## Usage

``` r
set_horseshoe(local_sparsity = 1, group_sparsity = 1, global_sparsity = 1)

# S3 method for class 'horseshoespec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.horseshoespec(x)

# S3 method for class 'horseshoespec'
knit_print(x, ...)
```

## Arguments

- local_sparsity:

  Initial local shrinkage hyperparameters

- group_sparsity:

  Initial group shrinkage hyperparameters

- global_sparsity:

  Initial global shrinkage hyperparameter

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Details

Set horseshoe prior initialization for VAR family.

- `local_sparsity`: Initial local shrinkage

- `group_sparsity`: Initial group shrinkage

- `global_sparsity`: Initial global shrinkage

In this package, horseshoe prior model is estimated by Gibbs sampling,
initial means initial values for that gibbs sampler.

## References

Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
estimator for sparse signals. Biometrika, 97(2), 465-480.

Makalic, E., & Schmidt, D. F. (2016). *A Simple Sampler for the
Horseshoe Estimator*. IEEE Signal Processing Letters, 23(1), 179-182.
