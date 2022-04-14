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
#' [etf_vix] and [oxfordman] range from 2012-01-09 to 2015-06-27 (only weekdays).
#' Each year corresponds to Italian default and Grexit.
#' If you wonder the exact vector of the date, see [trading_day] vector.
#' 
#' @keywords internal
#' @name financial_history_appendix
NULL

#' @rdname financial_history_appendix
#' 
#' @format A vector `trading_day` saves dates of each [etf_vix] and [oxfordman].
"trading_day"

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
#' Source: \url{https://www.cboe.com}
#' 
#' Release: \url{https://www.cboe.com/us/options/market_statistics/daily/}
"etf_vix"

#' Oxford-Man Institute Realized Library
#' 
#' The realized measure of financial assets dataset provided by [Oxford-man Institute of Quantitative Finance](https://www.oxford-man.ox.ac.uk).
#' 
#' @details 
#' * As a raw dataset, we have internal dataset of long format `oxfordman_long`. It contains every realized measure.
#' * Denote that non-trading dates are excluded in `oxfordman_long`, not in `NA`. So be careful when dealing this set directly.
#' * For analysis, we widened the data for 5-min realized volatility (`rv5`) and realized kernel variance (`rk_parzen`), respectively.
#'     * `oxfordman_wide_rv`
#'     * `oxfordman_wide_rk`
#' * `oxford_rv` and `oxford_rk` are the sets whose `NA` values interpolated using [imputeTS::na_interpolation()].
#' * First three datasets should be called using [data()] function: `data(..., package = "bvhar")`.
#' * Only `oxford_rv` and `oxford_rk` is lazy loaded.
#' 
#' @format `oxfordman_long` is the raw data frame of 53507 rows and 20 columns (You cannot call this dataset.):
#' 
#' \describe{
#'     \item{date}{Date - From 2012-01-09 to 2015-06-27}
#'     \item{Symbol}{Name of the Assets - See below for each name}
#'     \item{nobs}{Number of observations}
#'     \item{by_ss}{Bipower Variation (5-min Sub-sampled)}
#'     \item{rsv}{Realized Semi-variance (5-min)}
#'     \item{rk_parzen}{Realized Kernel Variance (Non-Flat Parzen)}
#'     \item{rv10}{Realized Variance (10-min)}
#'     \item{rv5_ss}{Realized Variance (5-min Sub-sampled)}
#'     \item{rv5}{Realized Variance (5-min)}
#'     \item{rv10_ss}{Realized Variance (10-min Sub-sampled)}
#'     \item{rk_twoscale}{Realized Kernel Variance (Two-Scale/Bartlett)}
#'     \item{close_price}{Closing (Last) Price}
#'     \item{rsv_ss}{Realized Semi-variance (5-min Sub-sampled)}
#'     \item{rk_th2}{Realized Kernel Variance (Tukey-Hanning(2))}
#'     \item{open_time}{Opening Time}
#'     \item{medrv}{Median Realized Variance (5-min)}
#'     \item{open_price}{Opening (First) Price}
#'     \item{bv}{Bipower Variation (5-min)}
#'     \item{open_to_close}{Open to Close Return}
#'     \item{close_time}{Closing Time}
#' }
#' 
#' `oxfordman_rv` is a data frame that interpolates `NA` values of `oxfordman_wide_rv`.
#' Also, it does not have `date` column for fitting.
#' The number of rows is 905 and the number of columns is 30 (except date).
#' \describe{
#'     \item{date}{Date - From 2012-01-09 to 2015-06-27}
#'     \item{AEX}{AEX index}
#'     \item{AORD}{All Ordinaries}
#'     \item{BFX}{Bell 20 Index}
#'     \item{BSESN}{S&P BSE Sensex}
#'     \item{BVLG}{PSI All-Share Index (excluded because this index is observed from 2012-10-15)}
#'     \item{BVSP}{BVSP BOVESPA Index}
#'     \item{DJI}{Dow Jones Industrial Average}
#'     \item{FCHI}{CAC 40}
#'     \item{FTMIB}{FTSE MIB}
#'     \item{FTSE}{FTSE 100}
#'     \item{GDAXI}{DAX}
#'     \item{GSPTSE}{S&P/TSX Composite index}
#'     \item{HSI}{HANG SENG Index}
#'     \item{IBEX}{IBEX 35 Index}
#'     \item{IXIC}{Nasdaq 100}
#'     \item{KS11}{Korea Composite Stock Price Index (KOSPI)}
#'     \item{KSE}{Karachi SE 100 Index}
#'     \item{MXX}{IPC Mexico}
#'     \item{N225}{Nikkei 225}
#'     \item{NSEI}{NIFTY 50}
#'     \item{OMXC20}{OMX Copenhagen 20 Index}
#'     \item{OMXHPI}{OMX Helsinki All Share Index}
#'     \item{OMXSPI}{OMX Stockholm All Share Index}
#'     \item{OSEAX}{Oslo Exchange All-share Index}
#'     \item{RUT}{Russel 2000}
#'     \item{SMSI}{Madrid General Index}
#'     \item{SPX}{S&P 500 Index}
#'     \item{SSEC}{Shanghai Composite Index}
#'     \item{SSMI}{Swiss Stock Market Index}
#'     \item{STI}{Straits Times Index (excluded because this index is NA in the period)}
#'     \item{STOXX50E}{EURO STOXX 50}
#' }
#' 
#' @source 
#' [https://realized.oxford-man.ox.ac.uk/data](https://realized.oxford-man.ox.ac.uk/data)
#' 
#' Available estimators: [https://realized.oxford-man.ox.ac.uk/documentation/estimators](https://realized.oxford-man.ox.ac.uk/documentation/estimators)
#' 
#' Asset lists: [https://realized.oxford-man.ox.ac.uk/data/assets](https://realized.oxford-man.ox.ac.uk/data/assets)
#' 
#' @name oxfordman
"oxfordman_rv"

#' @rdname oxfordman
#' @format `oxfordman_rk` is a data frame that interpolates `NA` values of `oxfordman_wide_rk`.
#' Also, it does not have `DATE` column for fitting.
#' The number of rows is 1826 and the number of columns is 31.
"oxfordman_rk"
