# Density Plot for Minnesota Prior VAR Model

This function draws density plot for coefficient matrices of Minnesota
prior VAR model.

## Usage

``` r
# S3 method for class 'summary.normaliw'
autoplot(
  object,
  type = c("trace", "dens", "area"),
  pars = character(),
  regex_pars = character(),
  ...
)
```

## Arguments

- object:

  A `summary.normaliw` object

- type:

  The type of the plot. Trace plot (`trace`), kernel density plot
  (`dens`), and interval estimates plot (`area`).

- pars:

  Parameter names to draw.

- regex_pars:

  Regular expression parameter names to draw.

- ...:

  Other options for each
  [`bayesplot::mcmc_trace()`](https://mc-stan.org/bayesplot/reference/MCMC-traces.html),
  [`bayesplot::mcmc_dens()`](https://mc-stan.org/bayesplot/reference/MCMC-distributions.html),
  and
  [`bayesplot::mcmc_areas()`](https://mc-stan.org/bayesplot/reference/MCMC-intervals.html).

## Value

A ggplot object
