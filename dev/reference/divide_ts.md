# Split a Time Series Dataset into Train-Test Set

Split a given time series dataset into train and test set for
evaluation.

## Usage

``` r
divide_ts(y, n_ahead)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- n_ahead:

  step to evaluate

## Value

List of two datasets, train and test.
