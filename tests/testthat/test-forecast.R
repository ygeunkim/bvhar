# VAR----------------------------------
test_that("Test for varlse forecast", {
  fit_var <- var_lm(etf_vix, 5)
  pred_var <- predict(fit_var, 10)
  
  expect_s3_class(pred_var, "predbvhar")
  
  
})
#> Test passed 🌈