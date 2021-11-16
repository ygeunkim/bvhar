## code to prepare `etf-vix` dataset goes here
library(readr)
library(dplyr)
library(purrr)
# lists of files------------------------------
file_list <- list.files(
  "data-raw/etf",
  full.names = TRUE
)
data_list <- lapply(file_list, read_csv, na = c("", "NA", "."))
# DATE, variables-----------------------------
etf_vix_raw <- reduce(data_list, left_join, by = "DATE")
# only variables and impute missing-----------
etf_vix <- 
  etf_vix_raw %>% 
  select(-DATE) %>% 
  apply(2, imputeTS::na_interpolation) %>% 
  as_tibble()

# usethis::use_data(etf_vix_raw, internal = TRUE, overwrite = TRUE)
usethis::use_data(etf_vix, overwrite = TRUE)
