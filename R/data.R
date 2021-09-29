#' CBOE ETF Volatility Index Dataset
#' 
#' Chicago Board Options Exchage (CBOE) Exchange Traded Funds (ETFs) volatility index from FRED.
#' 
#' @details 
#' Copyright, 2016, Chicago Board Options Exchange, Inc.
#' 
#' Note that, in this data frame, dates column is removed.
#' This dataset interpolated 36 missing observations (nontrading dates) using [imputeTS::na_interpolation()].
#' 
#' @format A data frame of 1006 row and 9 columns:
#' 
#' From 2015-01-05 to 2018-12-28,
#' 36 missing observations were interpolated by [stats::approx()] with `linear`.
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
#' Source: \url{https://www.cboe.com}
#' 
#' Release: \url{https://www.cboe.com/us/options/market_statistics/daily/}
"etf_vix"

#' CBOE ETF Volatility Index Raw Dataset
#' 
#' Raw dataset for Chicago Board Options Exchage (CBOE) Exchange Traded Funds (ETFs) volatility index from FRED.
#' 
#' @details 
#' This dataset is included for convenience of usage of the CBOE ETF dataset when academic researching.
#' For the details of the data, see [etf_vix] documentation.
#' 
#' @format A data frmae of 1006 row and 10 columns,
#' including date column (`DATE`)
#' 
#' From 2015-01-05 to 2018-12-28,
#' there exists 36 missing obervations.
#' 
#' @source 
#' Source: \url{https://www.cboe.com}
#' 
#' Release: \url{https://www.cboe.com/us/options/market_statistics/daily/}
"etf_vix_raw"
