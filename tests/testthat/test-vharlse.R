# Components of vharlse--------------
test_that("Test for vharlse class", {
  fit_test_vhar <- vhar_lm(etf_vix)
  num_col <- ncol(etf_vix)
  num_row <- nrow(etf_vix)
  
  expect_s3_class(fit_test_vhar, "vharlse")
  
  expect_equal(
    nrow(fit_test_vhar$coef), 
    fit_test_vhar$p * ncol(etf_vix) + 1
  )
  
  expect_equal(
    nrow(fit_test_vhar$coef), 
    ifelse(fit_test_vhar$type == "none", fit_test_vhar$p * num_col, fit_test_vhar$p * num_col + 1)
  )
  
  expect_equal(
    nrow(fit_test_vhar$HARtrans),
    ifelse(fit_test_vhar$type == "none", fit_test_vhar$p * num_col, fit_test_vhar$p * num_col + 1)
  )
  
  expect_equal(
    ncol(fit_test_vhar$HARtrans),
    ifelse(fit_test_vhar$type == "none", fit_test_vhar$month * num_col, fit_test_vhar$month * num_col + 1)
  )
  
  expect_equal(
    ncol(fit_test_vhar$coef),
    num_col
  )
})
#> Test passed 🌈