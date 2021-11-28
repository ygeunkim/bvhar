# Components of vharlse--------------
test_that("Test for vharlse class", {
  fit_test_vhar <- vhar_lm(etf_vix)
  
  expect_s3_class(fit_test_vhar, "vharlse")
  
  expect_equal(
    nrow(fit_test_vhar$coef), 
    3 * ncol(etf_vix) + 1
  )
  expect_equal(
    ncol(fit_test_vhar$coef),
    ncol(etf_vix)
  )
})
#> Test passed 🌈