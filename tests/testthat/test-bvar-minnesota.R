# Components of bvarmn--------------
test_that("Test for bvarmn class", {
  skip_on_cran()
  
  test_lag <- 3
  etf_ncol <- ncol(etf_vix[1:50, 1:3])
  etf_spec <- set_bvar(
    sigma = apply(etf_vix[1:50, 1:3], 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvar <- bvar_minnesota(
    y = etf_vix[1:50, 1:3],
    p = test_lag,
    bayes_spec = etf_spec
  )
  
  expect_s3_class(etf_spec, "bvharspec")
  expect_s3_class(fit_test_bvar, "bvarmn")
  
  expect_equal(
    nrow(fit_test_bvar$coef), 
    ncol(etf_vix[1:50, 1:3]) * test_lag + 1
  )
  expect_equal(
    ncol(fit_test_bvar$coef),
    ncol(etf_vix[1:50, 1:3])
  )
  
})
#> Test passed 🌈