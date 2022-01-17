test_that("loglikelihood", {
  test_lag <- 3
  fit_test_var <- var_lm(etf_vix, test_lag)
  
  fit_test_vhar <- vhar_lm(etf_vix)
  
  etf_ncol <- ncol(etf_vix)
  bvar_spec <- set_bvar(
    sigma = apply(etf_vix, 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvar <- bvar_minnesota(etf_vix, test_lag, bvar_spec)
  
  bvhar_spec <- set_bvhar(
    sigma = apply(etf_vix, 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvhar <- bvhar_minnesota(etf_vix, bvhar_spec)
  
  expect_s3_class(logLik(fit_test_var), "logLik")
  expect_s3_class(logLik(fit_test_vhar), "logLik")
  expect_s3_class(logLik(fit_test_bvar), "logLik")
  expect_s3_class(logLik(fit_test_bvhar), "logLik")
  
})
