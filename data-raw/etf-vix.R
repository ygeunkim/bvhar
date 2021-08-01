## code to prepare `etf-vix` dataset goes here
library(readr)
library(dplyr)
library(purrr)
# lists of files---------
file_list <- list.files(
  "data-raw/etf",
  full.names = TRUE
)
data_list <- lapply(file_list, read_csv, na = c("", "NA", "."))
etf_vix <- 
  reduce(data_list, left_join, by = "DATE") %>% 
  select(-DATE) %>% 
  na.omit()

usethis::use_data(etf_vix, overwrite = TRUE)
