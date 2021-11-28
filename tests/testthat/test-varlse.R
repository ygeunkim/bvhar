# Components of varlse--------------
test_that("Test for varlse class", {
  test_lag <- 3
  fit_test_var <- var_lm(etf_vix, test_lag)
  
  expect_s3_class(fit_test_var, "varlse")
  
  expect_equal(
    nrow(fit_test_var$coef), 
    ncol(etf_vix) * test_lag + 1
  )
  expect_equal(
    ncol(fit_test_var$coef),
    ncol(etf_vix)
  )
})
#> Test passed 🌈