## code to prepare `oxfordman_long` dataset goes here
library(readr)
library(dplyr)
oxfordman_long <- read_csv("data-raw/oxfordmanrealizedvolatilityindices.csv")
names(oxfordman_long)[1] <- "DATE"
# pre-processing------------------------------------
oxfordman_long <- 
  oxfordman_long %>% 
  mutate(
    DATE = lubridate::as_date(DATE),
    Symbol = stringr::str_remove(Symbol, pattern = "\\.")
  ) %>% # remove the dot in front of each name of the asset (Symbol)
  filter(between(
    DATE,
    lubridate::as_date("2013-01-05"),
    lubridate::as_date("2019-12-28")
  )) # filtering dates
# Widen data----------------------------------------
spread_oxford <- function(x = oxfordman_long, var = "rv5") {
  rv <- sym(var)
  x %>% 
    mutate(realized = !!rv) %>% 
    select(DATE, Symbol, realized) %>% 
    filter(Symbol != "STI") %>% # STI has too many NAs
    tidyr::pivot_wider(names_from = "Symbol", values_from = "realized") %>% 
    arrange(DATE)
}
# 5-min RV------------------------------------------
oxfordman_wide_rv <- spread_oxford(oxfordman_long, "rv5")
# Realized Kernel Variance (Non-Flat Parzen)--------
oxfordman_wide_rk <- spread_oxford(oxfordman_long, "rk_parzen")
# Add holidays as NA rows---------------------------
# DAY <- lubridate::as_date(oxfordman_wide_rv$DATE)
# holiday <- which(diff(DAY) != 1 & diff(DAY) != 3)
# row_na <- 
#   holiday %>% 
#   sapply(
#     function(x) {
#       seq(
#         from = DAY[x] + lubridate::days(1),
#         to = DAY[x + 1] - lubridate::days(1),
#         by = "day"
#       )
#     }
#   )
# Function to add NA-------------------------------
# oxfordman_rk %>% 
#   add_row(DATE = row_na[[1]], .after = holiday[1])
# Impute-------------------------------------------
oxfordman_rv <- 
  oxfordman_wide_rv %>% 
  select(-DATE) %>% 
  apply(2, imputeTS::na_interpolation) %>% 
  as_tibble()
oxfordman_rk <- 
  oxfordman_wide_rk %>% 
  select(-DATE) %>% 
  apply(2, imputeTS::na_interpolation) %>% 
  as_tibble()

usethis::use_data(oxfordman_long, overwrite = TRUE)
usethis::use_data(oxfordman_wide_rv, overwrite = TRUE)
usethis::use_data(oxfordman_wide_rk, overwrite = TRUE)
usethis::use_data(oxfordman_rv, overwrite = TRUE)
usethis::use_data(oxfordman_rk, overwrite = TRUE)
