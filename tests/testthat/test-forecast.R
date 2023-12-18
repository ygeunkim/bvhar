# VAR----------------------------------
test_that("Test for varlse forecast", {
  skip_on_cran()
  
  num_col <- 3
  fit_var <- var_lm(etf_vix[, 1:3], 2)
  fit_vhar <- vhar_lm(etf_vix[, 1:3])
  
  num_forecast <- 2
  pred_var <- predict(fit_var, num_forecast)
  pred_vhar <- predict(fit_vhar, num_forecast)
  
  expect_s3_class(pred_var, "predbvhar")
  expect_s3_class(pred_vhar, "predbvhar")

  expect_equal(
    nrow(pred_var$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_var$forecast),
    num_col
  )
  expect_equal(
    nrow(pred_vhar$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_vhar$forecast),
    num_col
  )
  
})
#> Test passed 🌈