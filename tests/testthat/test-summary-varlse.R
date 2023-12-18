test_that("VAR Coefficients Summary Table", {
  skip_on_cran()
  
  test_lag <- 3
  num_col <- 2
  fit_test_var <- var_lm(etf_vix[, seq_len(num_col)], test_lag)
  fit_var_summary <- summary(fit_test_var)
  term_summary <- fit_var_summary$coefficients[, "term"]
  
  expect_equal(
    gsub(pattern = "\\..*", replacement = "", term_summary)[seq_len(fit_test_var$df)],
    rownames(fit_test_var$coefficients)
  )
  
  expect_equal(
    unique(gsub(pattern = ".*\\.", replacement = "", term_summary)),
    colnames(fit_test_var$coefficients)
  )
})
