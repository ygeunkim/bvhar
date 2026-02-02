# Convert VAR to VMA(infinite)

Convert VAR process to infinite vector MA process

## Usage

``` r
VARtoVMA(object, lag_max)
```

## Arguments

- object:

  A `varlse` object

- lag_max:

  Maximum lag for VMA

## Value

VMA coefficient of k(lag-max + 1) x k dimension

## Details

Let VAR(p) be stable. \$\$Y_t = c + \sum\_{j = 0} W_j Z\_{t - j}\$\$ For
VAR coefficient \\B_1, B_2, \ldots, B_p\\, \$\$I = (W_0 + W_1 L + W_2
L^2 + \cdots + ) (I - B_1 L - B_2 L^2 - \cdots - B_p L^p)\$\$
Recursively, \$\$W_0 = I\$\$ \$\$W_1 = W_0 B_1 (W_1^T = B_1^T W_0^T)\$\$
\$\$W_2 = W_1 B_1 + W_0 B_2 (W_2^T = B_1^T W_1^T + B_2^T W_0^T)\$\$
\$\$W_j = \sum\_{j = 1}^k W\_{k - j} B_j (W_j^T = \sum\_{j = 1}^k B_j^T
W\_{k - j}^T)\$\$

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.
