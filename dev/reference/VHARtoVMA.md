# Convert VHAR to VMA(infinite)

Convert VHAR process to infinite vector MA process

## Usage

``` r
VHARtoVMA(object, lag_max)
```

## Arguments

- object:

  A `vharlse` object

- lag_max:

  Maximum lag for VMA

## Value

VMA coefficient of k(lag-max + 1) x k dimension

## Details

Let VAR(p) be stable and let VAR(p) be \\Y_0 = X_0 B + Z\\

VHAR is VAR(22) with \$\$Y_0 = X_1 B + Z = ((X_0 \tilde{T}^T)) \Phi +
Z\$\$

Observe that \$\$B = \tilde{T}^T \Phi\$\$

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.
