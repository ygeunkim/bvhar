# Compare Lists of Models

Draw plot of test error for given models

## Usage

``` r
gg_loss(
  mod_list,
  y,
  type = c("mse", "mae", "mape", "mase"),
  mean_line = FALSE,
  line_param = list(),
  mean_param = list(),
  viridis = FALSE,
  viridis_option = "D",
  NROW = NULL,
  NCOL = NULL,
  ...
)
```

## Arguments

- mod_list:

  Lists of forecast results (`predbvhar` objects)

- y:

  Test data to be compared. should be the same format with the train
  data and predict\$forecast.

- type:

  Loss function to be used (`mse`: MSE, `mae`: MAE, `mape`: MAPE,
  `mase`: MASE)

- mean_line:

  Whether to draw average loss. By default, `FALSE`.

- line_param:

  Parameter lists for
  [`ggplot2::geom_path()`](https://ggplot2.tidyverse.org/reference/geom_path.html).

- mean_param:

  Parameter lists for average loss with
  [`ggplot2::geom_hline()`](https://ggplot2.tidyverse.org/reference/geom_abline.html).

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

  Additional options for `geom_loss` (`inherit.aes` and `show.legend`)

## Value

A ggplot object

## See also

- [`mse()`](mse.md) to compute MSE for given forecast result

- [`mae()`](mae.md) to compute MAE for given forecast result

- [`mape()`](mape.md) to compute MAPE for given forecast result

- [`mase()`](mase.md) to compute MASE for given forecast result
