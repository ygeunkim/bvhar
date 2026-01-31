# Dynamic Spillover Indices Plot

Draws dynamic directional spillover plot.

## Usage

``` r
# S3 method for class 'bvhardynsp'
autoplot(
  object,
  type = c("tot", "to", "from", "net"),
  hcol = "grey",
  hsize = 1.5,
  row_facet = NULL,
  col_facet = NULL,
  ...
)
```

## Arguments

- object:

  A `bvhardynsp` object

- type:

  Index to draw

- hcol:

  color of horizontal line = 0 (By default, grey)

- hsize:

  size of horizontal line = 0 (By default, 1.5)

- row_facet:

  `nrow` of
  [`ggplot2::facet_wrap()`](https://ggplot2.tidyverse.org/reference/facet_wrap.html)

- col_facet:

  `ncol` of
  [`ggplot2::facet_wrap()`](https://ggplot2.tidyverse.org/reference/facet_wrap.html)

- ...:

  Additional
