# Components of bvarflat--------------
test_that("Test for bvarflat class", {
  skip_on_cran()
  
  test_lag <- 3
  etf_spec <- set_bvar_flat()
  fit_test_bvar <- bvar_flat(etf_vix, test_lag, bayes_spec = etf_spec)
  
  expect_s3_class(etf_spec, "bvharspec")
  expect_s3_class(fit_test_bvar, "bvarflat")
  
  expect_equal(
    nrow(fit_test_bvar$coef), 
    ncol(etf_vix) * test_lag + 1
  )
  expect_equal(
    ncol(fit_test_bvar$coef),
    ncol(etf_vix)
  )
  
})
#> Test passed 🌈