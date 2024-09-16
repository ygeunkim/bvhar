#' Time points and Financial Events
#' 
#' @description 
#' This page describes about some important financial events in 20th century.
#' This might give some hint when cutting data and why we provides datasets in limited period.
#' 
#' # Outline
#' 
#' * 2000: Dot-com bubble
#' * 2001: September 11 terror and Enron scandal
#' * 2003: Iraq war (until 2011)
#' * 2007 to 2008: Financial crisis (US)
#'     * 2007: Subprime morgage crisis
#'     * 2008: Bankrupcy of Lehman Brothers
#' * 2010 to 2016: European sovereign dept crisis
#'     * 2010: Greek debt crisis
#'     * 2011: Italian default
#'     * 2015: Greek default
#'     * 2016: Brexit
#' * 2018: US-China trade war
#' * 2019: Brexit
#' * 2020: COVID-19
#' 
#' # About Datasets in this package
#' 
#' [etf_vix] ranges from 2012-01-09 to 2015-06-27 (only weekdays).
#' Each year corresponds to Italian default and Grexit.
#' If you wonder the exact vector of the date, see [trading_day] vector.
#' 
#' # Notice
#' 
#' If you want other time period, see codes in the Github repo for the dataset: [ygeunkim/bvhar/data-raw/etf_vix.R](https://github.com/ygeunkim/bvhar/blob/master/data-raw/etf_vix.R)
#' 
#' You can download what you want by changing a few lines.
#' 
#' @keywords internal
#' @name financial_history_appendix
NULL

#' @rdname financial_history_appendix
#' 
#' @format A vector `trading_day` saves dates of [etf_vix].
"trading_day"

#' CBOE ETF Volatility Index Dataset
#' 
#' Chicago Board Options Exchage (CBOE) Exchange Traded Funds (ETFs) volatility index from FRED.
#' 
#' @details 
#' Copyright, 2016, Chicago Board Options Exchange, Inc.
#' 
#' Note that, in this data frame, dates column is removed.
#' This dataset interpolated 36 missing observations (nontrading dates) using `imputeTS::na_interpolation()`.
#' 
#' @format A data frame of 1006 row and 9 columns:
#' 
#' From 2012-01-09 to 2015-06-27,
#' 33 missing observations were interpolated by [stats::approx()] with `linear`.
#' \describe{
#'     \item{GVZCLS}{Gold ETF volatility index}
#'     \item{VXFXICLS}{China ETF volatility index}
#'     \item{OVXCLS}{Crude Oil ETF volatility index}
#'     \item{VXEEMCLS}{Emerging Markets ETF volatility index}
#'     \item{EVZCLS}{EuroCurrency ETF volatility index}
#'     \item{VXSLVCLS}{Silver ETF volatility index}
#'     \item{VXGDXCLS}{Gold Miners ETF volatility index}
#'     \item{VXXLECLS}{Energy Sector ETF volatility index}
#'     \item{VXEWZCLS}{Brazil ETF volatility index}
#' }
#' 
#' @references 
#' Chicago Board Options Exchange, CBOE Gold ETF Volatility Index (GVZCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/GVZCLS](https://fred.stlouisfed.org/series/GVZCLS), July 31, 2021.
#' 
#' Chicago Board Options Exchange, CBOE China ETF Volatility Index (VXFXICLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXFXICLS](https://fred.stlouisfed.org/series/VXFXICLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Crude Oil ETF Volatility Index (OVXCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/OVXCLS](https://fred.stlouisfed.org/series/OVXCLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Emerging Markets ETF Volatility Index (VXEEMCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXEEMCLS](https://fred.stlouisfed.org/series/VXEEMCLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE EuroCurrency ETF Volatility Index (EVZCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/EVZCLS](https://fred.stlouisfed.org/series/EVZCLS), August 2, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Silver ETF Volatility Index (VXSLVCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXSLVCLS](https://fred.stlouisfed.org/series/VXSLVCLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Gold Miners ETF Volatility Index (VXGDXCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXGDXCLS](https://fred.stlouisfed.org/series/VXGDXCLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Energy Sector ETF Volatility Index (VXXLECLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXXLECLS](https://fred.stlouisfed.org/series/VXXLECLS), August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Brazil ETF Volatility Index (VXEWZCLS), retrieved from FRED, Federal Reserve Bank of St. Louis; [https://fred.stlouisfed.org/series/VXEWZCLS](https://fred.stlouisfed.org/series/VXEWZCLS), August 2, 2021.
#' 
#' @source 
#' Source: [https://www.cboe.com](https://www.cboe.com)
#' 
#' Release: [https://www.cboe.com/us/options/market_statistics/daily/](https://www.cboe.com/us/options/market_statistics/daily/)
"etf_vix"
