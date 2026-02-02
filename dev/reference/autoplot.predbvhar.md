# Plot Forecast Result

Plots the forecasting result with forecast regions.

## Usage

``` r
# S3 method for class 'predbvhar'
autoplot(
  object,
  type = c("grid", "wrap"),
  ci_alpha = 0.7,
  alpha_scale = 0.3,
  x_cut = 1,
  viridis = FALSE,
  viridis_option = "D",
  NROW = NULL,
  NCOL = NULL,
  ...
)

# S3 method for class 'predbvhar'
autolayer(object, ci_fill = "grey70", ci_alpha = 0.5, alpha_scale = 0.3, ...)
```

## Arguments

- object:

  A `predbvhar` object

- type:

  Divide variables using
  [`ggplot2::facet_grid()`](https://ggplot2.tidyverse.org/reference/facet_grid.html)
  ("grid": default) or
  [`ggplot2::facet_wrap()`](https://ggplot2.tidyverse.org/reference/facet_wrap.html)
  ("wrap")

- ci_alpha:

  Transparency of CI

- alpha_scale:

  Scale of transparency parameter (`alpha`) between the two layers.
  `alpha` of CI ribbon = `alpha_scale` \* `alpha` of path (By default,
  .5)

- x_cut:

  plot x axes from `x_cut` for visibility

- viridis:

  If `TRUE`, scale CI and forecast line using
  [`ggplot2::scale_fill_viridis_d()`](https://ggplot2.tidyverse.org/reference/scale_viridis.html)
  and ggplot2::scale_colour_viridis_d, respectively.

- viridis_option:

  Option for viridis string. See `option` of
  ggplot2::scale_colour_viridis_d. Choose one of
  `c("A", "B", "C", "D", "E")`. By default, `D`.

- NROW:

  `nrow` of
  [`ggplot2::facet_wrap()`](https://ggplot2.tidyverse.org/reference/facet_wrap.html)

- NCOL:

  `ncol` of
  [`ggplot2::facet_wrap()`](https://ggplot2.tidyverse.org/reference/facet_wrap.html)

- ...:

  additional option for
  [`ggplot2::geom_path()`](https://ggplot2.tidyverse.org/reference/geom_path.html)

- ci_fill:

  color of CI

## Value

A ggplot object

A ggplot layer
