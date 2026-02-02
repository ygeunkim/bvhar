# Stability of the process

Check the stability condition of coefficient matrix.

## Usage

``` r
is.stable(x, ...)

# S3 method for class 'varlse'
is.stable(x, ...)

# S3 method for class 'vharlse'
is.stable(x, ...)

# S3 method for class 'bvarmn'
is.stable(x, ...)

# S3 method for class 'bvarflat'
is.stable(x, ...)

# S3 method for class 'bvharmn'
is.stable(x, ...)
```

## Arguments

- x:

  Model fit

- ...:

  not used

## Value

logical class

logical class

## Details

VAR(p) is stable if

\$\$\det(I_m - A z) \neq 0\$\$

for \\\lvert z \rvert \le 1\\.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.
