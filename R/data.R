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

#' @rdname etf_vix
#' @details 
#' On the other hand, you can call `etf_vix_raw` with [data()] function:
#' `data(etf_vix_raw, package = "bvhar")`.
#' @format `etf_vix_raw` is a raw dataset that includes date column (`DATE`).
#' 
#' From 2015-01-05 to 2018-12-28,
#' there exists 36 missing obervations.
"etf_vix_raw"

#' Oxford-Man Institute Realized Library
#' 
#' The realized measure of financial assets dataset provided by [Oxford-man Institute of Quantitative Finance](https://www.oxford-man.ox.ac.uk).
#' 
#' @details 
#' * As a raw dataset, we provide long format `oxfordman_long`. It contains every realized measure.
#' * Denote that non-trading dates are excluded in `oxfordman_long`, not in `NA`. So be careful when dealing this set directly.
#' * For analysis, we widened the data for 5-min realized volatility (`rv5`) and realized kernel variance (`rk_parzen`), respectively.
#'     * `oxfordman_wide_rv`
#'     * `oxfordman_wide_rk`
#' * `oxford_rv` and `oxford_rk` are the sets whose `NA` values interpolated using [imputeTS::na_interpolation()].
#' * First three datasets should be called using [data()] function: `data(..., package = "bvhar")`.
#' * Only `oxford_rv` and `oxford_rk` is lazy loaded.
#' 
#' @format `oxfordman_long` is the raw data frame of 53507 rows and 20 columns:
#' 
#' \describe{
#'     \item{DATE}{Date - From 2013-01-07 to 2019-12-27}
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
#' @source 
#' [https://realized.oxford-man.ox.ac.uk/data](https://realized.oxford-man.ox.ac.uk/data)
#' 
#' Available estimators: [https://realized.oxford-man.ox.ac.uk/documentation/estimators](https://realized.oxford-man.ox.ac.uk/documentation/estimators)
#' 
#' Asset lists: [https://realized.oxford-man.ox.ac.uk/data/assets](https://realized.oxford-man.ox.ac.uk/data/assets)
#' 
#' @name oxfordman
"oxfordman_long"

#' @rdname oxfordman
#' @format `oxfordman_wide_rv` is widened data frame of which values are 5-min RV (`rv5`).
#' The number of rows is 1826 and the number of columns is 31.
#' \describe{
#'     \item{DATE}{Date - From 2013-01-07 to 2019-12-27}
#'     \item{AEX}{AEX index}
#'     \item{AORD}{All Ordinaries}
#'     \item{BFX}{Bell 20 Index}
#'     \item{BSESN}{S&P BSE Sensex}
#'     \item{BVLG}{PSI All-Share Index}
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
#'     \item{STI}{Straits Times Index}
#'     \item{STOXX50E}{EURO STOXX 50}
#' }
"oxfordman_wide_rv"

#' @rdname oxfordman
#' @format `oxfordman_wide_rk` is widened data frame of which values are realized kernel variance (`rk_parzen`).
#' The number of rows is 1826 and the number of columns is 31, which are the same variables as `oxfordman_wide_rv`.
"oxfordman_wide_rk"

#' @rdname oxfordman
#' @format `oxfordman_rv` is a data frame that interpolates `NA` values of `oxfordman_wide_rv`.
#' Also, it does not have `DATE` column for fitting.
"oxfordman_rv"

#' @rdname oxfordman
#' @format `oxfordman_rk` is a data frame that interpolates `NA` values of `oxfordman_wide_rk`.
#' Also, it does not have `DATE` column for fitting.
"oxfordman_rk"

#' Estimated Energy Consumption
#' 
#' US estimated energy consumption (in Megawatts) in each region.
#' 
#' @format `est_energy` is realized kernel variance dataset computed by parzen kernel.
#' Since `NI` and `PJM_Load` observations are too old, they are excluded.
#' The observations are subset of the raw data, from 2014-08-04 (Monday) to 2018-08-03 (Friday).
#' \describe{
#'     \item{AEP}{\href{https://en.wikipedia.org/wiki/American_Electric_Power}{American Electric Power}}
#'     \item{COMED}{\href{https://en.wikipedia.org/wiki/Commonwealth_Edison}{Commonwealth Edison}}
#'     \item{DAYTON}{\href{https://en.wikipedia.org/wiki/DPL_Inc.}{The Dayton Power and Light Company}}
#'     \item{DEOK}{\href{https://en.wikipedia.org/wiki/Duke_Energy}{Duke Energy Ohio/Kentucky}}
#'     \item{DOM}{\href{https://en.wikipedia.org/wiki/Dominion_Energy}{Dominion Virginia Power}}
#'     \item{DUQ}{\href{https://en.wikipedia.org/wiki/DQE}{Duquesne Light Co.}}
#'     \item{EKPC}{\href{http://www.ekpc.coop/}{East Kentucky Power Cooperative}}
#'     \item{FE}{\href{https://en.wikipedia.org/wiki/FirstEnergy}{FirstEnergy}}
#'     \item{NI}{Northern Illinois Hub}
#'     \item{PJME}{PJM East Region: 2001-2018}
#'     \item{PJMW}{PJM West Region: 2001-2018}
#'     \item{PJM_Load}{PJM Load Combined: 1998-2001}
#' }
#' 
#' `est_energy_raw` is a raw dataset that contains hourly date and corresponding energy consumption.
#' \describe{
#'     \item{Datetime}{From 1998-04-01 10:00:00 to 2018-08-03 09:00:00}
#' }
#' 
#' `est_energy_rt` is a log-return dataset of `est_energy_raw`.
#' 
#' @source 
#' [https://www.kaggle.com/robikscube/time-series-forecasting-with-prophet/data](https://www.kaggle.com/robikscube/time-series-forecasting-with-prophet/data)
"est_energy"

