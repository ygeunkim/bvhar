# Time points and Financial Events

This page describes about some important financial events in 20th
century. This might give some hint when cutting data and why we provides
datasets in limited period.

## Usage

``` r
trading_day
```

## Format

A vector `trading_day` saves dates of [etf_vix](etf_vix.md).

## Outline

- 2000: Dot-com bubble

- 2001: September 11 terror and Enron scandal

- 2003: Iraq war (until 2011)

- 2007 to 2008: Financial crisis (US)

  - 2007: Subprime morgage crisis

  - 2008: Bankrupcy of Lehman Brothers

- 2010 to 2016: European sovereign dept crisis

  - 2010: Greek debt crisis

  - 2011: Italian default

  - 2015: Greek default

  - 2016: Brexit

- 2018: US-China trade war

- 2019: Brexit

- 2020: COVID-19

## About Datasets in this package

[etf_vix](etf_vix.md) ranges from 2012-01-09 to 2015-06-27 (only
weekdays). Each year corresponds to Italian default and Grexit. If you
wonder the exact vector of the date, see trading_day vector.

## Notice

If you want other time period, see codes in the Github repo for the
dataset:
[ygeunkim/bvhar/data-raw/etf_vix.R](https://github.com/ygeunkim/bvhar/blob/master/data-raw/etf_vix.R)

You can download what you want by changing a few lines.
