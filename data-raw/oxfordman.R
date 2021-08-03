## code to prepare `oxfordman` dataset goes here
library(readr)
# import---------------------------------------
oxfordman <- read_csv("data-raw/oxfordmanrealizedvolatilityindices.csv")

usethis::use_data(oxfordman, overwrite = TRUE)
