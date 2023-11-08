# Components of varlse--------------
test_that("Test for varlse class", {
  skip_on_cran()
  
  test_lag <- 3
  fit_test_var <- var_lm(etf_vix, test_lag)
  num_col <- ncol(etf_vix)
  num_row <- nrow(etf_vix)
  
  expect_s3_class(fit_test_var, "varlse")
  
  expect_equal(
    nrow(fit_test_var$coef), 
    ifelse(fit_test_var$type == "none", fit_test_var$p * num_col, fit_test_var$p * num_col + 1)
  )
  
  expect_equal(
    fit_test_var$df, 
    ifelse(fit_test_var$type == "none", fit_test_var$p * num_col, fit_test_var$p * num_col + 1)
  )
  
  expect_equal(
    nrow(fit_test_var$design),
    fit_test_var$obs
  )
  
  expect_equal(
    ncol(fit_test_var$y0),
    num_col
  )
  
  expect_equal(
    ncol(fit_test_var$coef),
    num_col
  )
})
#> Test passed 🌈