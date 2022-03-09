## code to prepare `oxfordman_long` dataset goes here
library(readr)
library(dplyr)
# Download from oxford-man site---------------------
temp_env <- tempfile()
download.file("https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip", temp_env)
oxfordman_long <- read_csv(unz(temp_env, "oxfordmanrealizedvolatilityindices.csv"))
names(oxfordman_long)[1] <- "date"
# pre-processing------------------------------------
oxfordman_long <- 
  oxfordman_long %>% 
  mutate(
    date = lubridate::as_date(date),
    Symbol = stringr::str_remove(Symbol, pattern = "\\.")
  ) %>% # remove the dot in front of each name of the asset (Symbol)
  filter(between(
    date,
    lubridate::as_date("2013-01-05"),
    lubridate::as_date("2019-12-28")
  )) # filtering dates
# Widen data----------------------------------------
spread_oxford <- function(x = oxfordman_long, var = "rv5") {
  rv <- sym(var)
  x %>% 
    mutate(realized = !!rv) %>% 
    select(date, Symbol, realized) %>% 
    filter(Symbol != "STI") %>% # STI has too many NAs
    tidyr::pivot_wider(names_from = "Symbol", values_from = "realized") %>% 
    arrange(date)
}
# 5-min RV------------------------------------------
oxfordman_wide_rv <- spread_oxford(oxfordman_long, "rv5")
# Realized Kernel Variance (Non-Flat Parzen)--------
oxfordman_wide_rk <- spread_oxford(oxfordman_long, "rk_parzen")
# Impute-------------------------------------------
oxfordman_rv <- 
  oxfordman_wide_rv %>% 
  select(-date) %>% 
  apply(2, imputeTS::na_interpolation) %>% 
  as_tibble()
oxfordman_rk <- 
  oxfordman_wide_rk %>% 
  select(-date) %>% 
  apply(2, imputeTS::na_interpolation) %>% 
  as_tibble()
# save----------------------------------------
usethis::use_data(oxfordman_rv, overwrite = TRUE)
usethis::use_data(oxfordman_rk, overwrite = TRUE)
