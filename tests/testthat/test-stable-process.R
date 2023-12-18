test_that("Stable root", {
  skip_on_cran()
  
  test_lag <- 3
  num_col <- 2
  fit_test_var <- var_lm(etf_vix[, seq_len(num_col)], test_lag)
  fit_test_vhar <- vhar_lm(etf_vix[, seq_len(num_col)])

  expect_equal(
    length(stableroot(fit_test_var)),
    num_col * fit_test_var$p
  )

  expect_equal(
    length(stableroot(fit_test_vhar)),
    num_col * fit_test_vhar$month
  )

})
