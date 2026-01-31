# Roots of characteristic polynomial

Compute the character polynomial of coefficient matrix.

## Usage

``` r
stableroot(x, ...)

# S3 method for class 'varlse'
stableroot(x, ...)

# S3 method for class 'vharlse'
stableroot(x, ...)

# S3 method for class 'bvarmn'
stableroot(x, ...)

# S3 method for class 'bvarflat'
stableroot(x, ...)

# S3 method for class 'bvharmn'
stableroot(x, ...)
```

## Arguments

- x:

  Model fit

- ...:

  not used

## Value

Numeric vector.

## Details

To know whether the process is stable or not, make characteristic
polynomial.

\$\$\det(I_m - A z) = 0\$\$

where \\A\\ is VAR(1) coefficient matrix representation.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.
