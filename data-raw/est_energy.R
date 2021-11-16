## code to prepare `est_energy` dataset goes here
library(dplyr)
library(highfrequency)
# Import parquet---------------------------------
est_energy_raw <- arrow::read_parquet("data-raw/est_hourly.paruqet")
# pre-process------------------------------------
# est_energy_raw %>% 
#   tidyr::pivot_longer(-Datetime, names_to = "name", values_to = "value") %>% 
#   filter(name == "NI", !is.na(value)) %>% 
#   summarise(range(Datetime))
# est_energy_raw %>% 
#   tidyr::pivot_longer(-Datetime, names_to = "name", values_to = "value") %>% 
#   filter(name == "PJM_Load", !is.na(value)) %>% 
#   summarise(range(Datetime))
# remove NI and PJM_Load-------------------------
est_energy_tmp <- 
  est_energy_raw %>% 
  filter(between(
    Datetime,
    lubridate::as_datetime("2014-08-04"), # Monday
    lubridate::as_datetime("2018-08-03") # Friday
  )) %>% 
  select(-NI, -PJM_Load) %>% 
  filter(!chron::is.weekend(Datetime)) %>% # use only weekdays
  arrange(Datetime)
# log-return-------------------------------------
compute_rt <- function(x) {
  log(x / lead(x))
}
est_energy_rt <- 
  est_energy_tmp %>% 
  mutate_if(
    is.numeric,
    list(~compute_rt(.))
  )
# High frequency---------------------------------
est_energy <- 
  est_energy_tmp %>% 
  rename(DT = Datetime) %>% 
  data.table::data.table() %>% 
  rKernelCov(rData = ., alignBy = "hour", kernelType = "parzen", makeReturns = TRUE)
num_dates <- length(est_energy)
freq_dates <- names(est_energy)[-num_dates]
est_energy <- 
  est_energy[seq_along(freq_dates)] %>% 
  lapply(diag) %>% 
  bind_rows()


usethis::use_data(est_energy, overwrite = TRUE)
# usethis::use_data(est_energy_raw, internal = TRUE, overwrite = TRUE)
# usethis::use_data(est_energy_rt, internal = TRUE, overwrite = TRUE)
