# Final Prediction Error Criterion

Compute FPE of VAR(p) and VHAR

## Usage

``` r
FPE(object, ...)

# S3 method for class 'varlse'
FPE(object, ...)

# S3 method for class 'vharlse'
FPE(object, ...)
```

## Arguments

- object:

  Model fit

- ...:

  not used

## Value

FPE value.

## Details

Let \\\tilde{\Sigma}\_e\\ be the MLE and let \\\hat{\Sigma}\_e\\ be the
unbiased estimator (`covmat`) for \\\Sigma_e\\. Note that

\$\$\tilde{\Sigma}\_e = \frac{n - k}{T} \hat{\Sigma}\_e\$\$

Then

\$\$FPE(p) = (\frac{n + k}{n - k})^m \det \tilde{\Sigma}\_e\$\$

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.
