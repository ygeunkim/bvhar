# VAR----------------------------------
test_that("Test for varlse forecast", {
  skip_on_cran()
  
  fit_var <- var_lm(etf_vix, 2)
  fit_vhar <- vhar_lm(etf_vix)
  
  pred_var <- predict(fit_var, 5)
  pred_vhar <- predict(fit_vhar, 5)
  
  expect_s3_class(pred_var, "predbvhar")
  expect_s3_class(pred_vhar, "predbvhar")
  
})
#> Test passed 🌈