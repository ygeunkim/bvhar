#' CBOE ETF Volatility Index
#' 
#' @description 
#' Chicago Board Options Exchage (CBOE) Exchange Traded Funds (ETFs) volatility index from FRED.
#' 
#' @details 
#' Copyright, 2016, Chicago Board Options Exchange, Inc.
#' 
#' Note that, in this data frame, dates column is removed.
#' 
#' @docType data
#' @format A data frame of 1006 row and 9 columns:
#' 
#' From 2015-01-02 to 2018-12-31,
#' excluding NA, i.e. non-trading dates.
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
#' @references 
#' Chicago Board Options Exchange, CBOE Gold ETF Volatility Index [GVZCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/GVZCLS}, July 31, 2021.
#' 
#' Chicago Board Options Exchange, CBOE China ETF Volatility Index [VXFXICLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXFXICLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Crude Oil ETF Volatility Index [OVXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/OVXCLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Emerging Markets ETF Volatility Index [VXEEMCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXEEMCLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE EuroCurrency ETF Volatility Index [EVZCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/EVZCLS}, August 2, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Silver ETF Volatility Index [VXSLVCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXSLVCLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Gold Miners ETF Volatility Index [VXGDXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXGDXCLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Energy Sector ETF Volatility Index [VXXLECLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXXLECLS}, August 1, 2021.
#' 
#' Chicago Board Options Exchange, CBOE Brazil ETF Volatility Index [VXEWZCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; \url{https://fred.stlouisfed.org/series/VXEWZCLS}, August 2, 2021.
#' 
"etf_vix"

#' Oxford-Man Institute Realized Library
#' 
#' @description 
#' Famous volatility measure dataset collected daily by \href{https://realized.oxford-man.ox.ac.uk}{Oxford-man Institute of Quantitative Finance}.
#' 
#' @docType data
#' @format A data frame of 53646 row and 20 columns:
#' \describe{
#'     \item{Date}{Date}
#'     \item{Symbol}{Name of the Assets}
#'     \itemize{
#'         \item{.AEX - AEX index}
#'         \item{.AORD - All Ordinaries}
#'         \item{.BFX -	Bell 20 Index}
#'         \item{.BSESN -	S&P BSE Sensex}
#'         \item{.BVLG - PSI All-Share Index}
#'         \item{.BVSP - BVSP BOVESPA Index}
#'         \item{.DJI - Dow Jones Industrial Average}
#'         \item{.FCHI - CAC 40}
#'         \item{.FTMIB - FTSE MIB}
#'         \item{.FTSE - FTSE 100}
#'         \item{.GDAXI - DAX}
#'         \item{.GSPTSE - S&P/TSX Composite index}
#'         \item{.HSI -	HANG SENG Index}
#'         \item{.IBEX - IBEX 35 Index}
#'         \item{.IXIC - Nasdaq 100}
#'         \item{.KS11 - Korea Composite Stock Price Index (KOSPI)}
#'         \item{.KSE - Karachi SE 100 Index}
#'         \item{.MXX - IPC Mexico}
#'         \item{.N225 - Nikkei 225}
#'         \item{.NSEI - NIFTY 50}
#'         \item{.OMXC20 - OMX Copenhagen 20 Index}
#'         \item{.OMXHPI - OMX Helsinki All Share Index}
#'         \item{.OMXSPI - OMX Stockholm All Share Index}
#'         \item{.OSEAX - Oslo Exchange All-share Index}
#'         \item{.RUT - Russel 2000}
#'         \item{.SMSI - Madrid General Index}
#'         \item{.SPX - S&P 500 Index}
#'         \item{.SSEC - Shanghai Composite Index}
#'         \item{.SSMI - Swiss Stock Market Index}
#'         \item{.STI - Straits Times Index}
#'         \item{.STOXX50E - EURO STOXX 50}
#'     }
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
#' @references 
#' \url{https://realized.oxford-man.ox.ac.uk/data}
#' 
#' Available estimators: \url{https://realized.oxford-man.ox.ac.uk/documentation/estimators}
#' 
#' Asset lists: \url{https://realized.oxford-man.ox.ac.uk/data/assets}
"oxfordman"
