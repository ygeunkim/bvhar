# Components of vharlse--------------
test_that("Test for vharlse class", {
  skip_on_cran()
  
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

test_that("Computation Methods", {
  skip_on_cran()
  
  fit_test_nor <- vhar_lm(etf_vix[, 1:3])
  fit_test_llt <- vhar_lm(etf_vix[, 1:3], method = "chol")
  fit_test_qr <- vhar_lm(etf_vix[, 1:3], method = "qr")

  expect_equal(
    fit_test_nor$coefficients,
    fit_test_llt$coefficients
  )

  expect_equal(
    fit_test_nor$coefficients,
    fit_test_qr$coefficients
  )
})
#> Test passed 🌈