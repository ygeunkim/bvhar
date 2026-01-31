# Plot the Result of BVAR and BVHAR MCMC

Draw BVAR and BVHAR MCMC plots.

## Usage

``` r
# S3 method for class 'bvharsp'
autoplot(
  object,
  type = c("coef", "trace", "dens", "area"),
  pars = character(),
  regex_pars = character(),
  ...
)
```

## Arguments

- object:

  A `bvharsp` object

- type:

  The type of the plot. Posterior coefficient (`coef`), Trace plot
  (`trace`), kernel density plot (`dens`), and interval estimates plot
  (`area`).

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
