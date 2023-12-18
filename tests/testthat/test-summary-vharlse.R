test_that("VHAR Coefficients Summary Table", {
  skip_on_cran()
  
  num_col <- 2
  fit_test_vhar <- vhar_lm(etf_vix[, seq_len(num_col)])
  fit_vhar_summary <- summary(fit_test_vhar)
  term_summary <- fit_vhar_summary$coefficients[, "term"]

  expect_equal(
    gsub(pattern = "\\..*", replacement = "", term_summary)[seq_len(fit_test_vhar$df)],
    rownames(fit_test_vhar$coefficients)
  )

  expect_equal(
    unique(gsub(pattern = ".*\\.", replacement = "", term_summary)),
    colnames(fit_test_vhar$coefficients)
  )
})
