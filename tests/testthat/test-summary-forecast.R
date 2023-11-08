# Train-test---------------------------
test_that("Test for data-splitting", {
  skip_on_cran()
  
  test_data <- matrix(rnorm(10100), ncol = 10, byrow = TRUE)
  n_test <- 10
  n_train <- nrow(test_data) - n_test
  data_split <- divide_ts(test_data, n_test)
  
  expect_equal(
    nrow(data_split$train), 
    n_train
  )
  expect_equal(
    nrow(data_split$test), 
    n_test
  )
})

test_that("Rolling windows", {
  skip_on_cran()
  
  etf_split <- divide_ts(etf_vix[1:100, 1:2], 10)
  etf_train <- etf_split$train
  etf_test <- etf_split$test
  
  fit_var <- var_lm(etf_train, 2)
  var_roll <- forecast_roll(fit_var, 10, etf_test)
  expect_s3_class(var_roll, "predbvhar_roll")
  expect_s3_class(var_roll, "bvharcv")
  
})

test_that("Expanding windows", {
  skip_on_cran()
  
  etf_split <- divide_ts(etf_vix[1:100, 1:2], 10)
  etf_train <- etf_split$train
  etf_test <- etf_split$test
  
  fit_var <- var_lm(etf_train, 2)
  var_expand <- forecast_expand(fit_var, 10, etf_test)
  expect_s3_class(var_expand, "predbvhar_expand")
  expect_s3_class(var_expand, "bvharcv")
  
})
#> Test passed 🌈