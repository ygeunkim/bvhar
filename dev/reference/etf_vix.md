# CBOE ETF Volatility Index Dataset

Chicago Board Options Exchage (CBOE) Exchange Traded Funds (ETFs)
volatility index from FRED.

## Usage

``` r
etf_vix
```

## Format

A data frame of 1006 row and 9 columns:

From 2012-01-09 to 2015-06-27, 33 missing observations were interpolated
by [`stats::approx()`](https://rdrr.io/r/stats/approxfun.html) with
`linear`.

- GVZCLS:

  Gold ETF volatility index

- VXFXICLS:

  China ETF volatility index

- OVXCLS:

  Crude Oil ETF volatility index

- VXEEMCLS:

  Emerging Markets ETF volatility index

- EVZCLS:

  EuroCurrency ETF volatility index

- VXSLVCLS:

  Silver ETF volatility index

- VXGDXCLS:

  Gold Miners ETF volatility index

- VXXLECLS:

  Energy Sector ETF volatility index

- VXEWZCLS:

  Brazil ETF volatility index

## Source

Source: <https://www.cboe.com>

Release: <https://www.cboe.com/us/options/market_statistics/daily/>

## Details

Copyright, 2016, Chicago Board Options Exchange, Inc.

Note that, in this data frame, dates column is removed. This dataset
interpolated 36 missing observations (nontrading dates) using
`imputeTS::na_interpolation()`.

## References

Chicago Board Options Exchange, CBOE Gold ETF Volatility Index (GVZCLS),
retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/GVZCLS>, July 31, 2021.

Chicago Board Options Exchange, CBOE China ETF Volatility Index
(VXFXICLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/VXFXICLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE Crude Oil ETF Volatility Index
(OVXCLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/OVXCLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE Emerging Markets ETF Volatility
Index (VXEEMCLS), retrieved from FRED, Federal Reserve Bank of St.
Louis; <https://fred.stlouisfed.org/series/VXEEMCLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE EuroCurrency ETF Volatility Index
(EVZCLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/EVZCLS>, August 2, 2021.

Chicago Board Options Exchange, CBOE Silver ETF Volatility Index
(VXSLVCLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/VXSLVCLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE Gold Miners ETF Volatility Index
(VXGDXCLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/VXGDXCLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE Energy Sector ETF Volatility Index
(VXXLECLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/VXXLECLS>, August 1, 2021.

Chicago Board Options Exchange, CBOE Brazil ETF Volatility Index
(VXEWZCLS), retrieved from FRED, Federal Reserve Bank of St. Louis;
<https://fred.stlouisfed.org/series/VXEWZCLS>, August 2, 2021.
