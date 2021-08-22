# Components of varlse--------------
test_that("Test for varlse class", {
  test_lag <- 3
  fit_test_var <- var_lm(etf_vix, test_lag)
  
  expect_s3_class(fit_test_var, "varlse")
  
  expect_equal(fit_test_var$p, test_lag)
  expect_equal(fit_test_var$m, ncol(etf_vix))
  expect_equal(fit_test_var$obs, nrow(etf_vix) - test_lag)
})
#> Test passed 🌈