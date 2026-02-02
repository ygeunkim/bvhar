# Setting Empirical Bayes Optimization Bounds

**\[deprecated\]** This function sets lower and upper bounds for
[`set_bvar()`](set_bvar.md), [`set_bvhar()`](set_bvar.md), or
[`set_weight_bvhar()`](set_bvar.md).

## Usage

``` r
bound_bvhar(
  init_spec = set_bvhar(),
  lower_spec = set_bvhar(),
  upper_spec = set_bvhar()
)

# S3 method for class 'boundbvharemp'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.boundbvharemp(x)

# S3 method for class 'boundbvharemp'
knit_print(x, ...)
```

## Arguments

- init_spec:

  Initial Bayes model specification

- lower_spec:

  Lower bound Bayes model specification

- upper_spec:

  Upper bound Bayes model specification

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

`boundbvharemp` [class](https://rdrr.io/r/base/class.html)
