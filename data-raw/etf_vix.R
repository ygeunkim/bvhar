## code to prepare `etf-vix` dataset goes here
library(readr)
library(dplyr)
library(fredr)
# FRED_API_KEY=fredapi in .Renviron file using usethis::edit_r_environ() function
# See http://sboysel.github.io/fredr/articles/fredr.html
fred_id <- fredr_series_search_text(search_text = "CBOE ETF")
# Import datasets-----------------------------
etf_vix_long <- purrr::map_dfr(
  fred_id$id, 
  fredr,
  observation_start = as.Date("2012-01-09"), # after Italian debt crisis
  observation_end = as.Date("2015-06-27") # before Grexit
) |> 
  select(date, series_id, value)
# date, variables-----------------------------
etf_vix_raw <- 
  etf_vix_long |> 
  tidyr::pivot_wider(names_from = "series_id", values_from = "value")
# only variables and impute missing-----------
etf_vix <- 
  etf_vix_raw |> 
  select(-date) |> 
  apply(2, imputeTS::na_interpolation) |> 
  as_tibble()
# only date-----------------------------------
trading_day <- etf_vix_raw$date
# save----------------------------------------
usethis::use_data(etf_vix, overwrite = TRUE)
usethis::use_data(trading_day, overwrite = TRUE)
