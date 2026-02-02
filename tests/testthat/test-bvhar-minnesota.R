# Components of var-type--------------
test_that("Test for VAR-type minnesota", {
  skip_on_cran()
  
  etf_ncol <- ncol(etf_vix[1:50, 1:3])
  var_spec <- set_bvhar(
    sigma = apply(etf_vix[1:50, 1:3], 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvhar_var <- bvhar_minnesota(
    y = etf_vix[1:50, 1:3],
    bayes_spec = var_spec
  )
  
  expect_s3_class(var_spec, "bvharspec")
  expect_s3_class(fit_test_bvhar_var, "bvharmn")
  
  expect_equal(
    nrow(fit_test_bvhar_var$coef), 
    ncol(etf_vix[1:50, 1:3]) * 3 + 1
  )
  expect_equal(
    ncol(fit_test_bvhar_var$coef),
    ncol(etf_vix[1:50, 1:3])
  )
  
})
# Components of vhar-type--------------
test_that("Test for VHAR-type minnesota", {
  skip_on_cran()
  
  etf_ncol <- ncol(etf_vix[1:50, 1:3])
  vhar_spec <- set_weight_bvhar(
    sigma = apply(etf_vix[1:50, 1:3], 2, sd),
    lambda = .2,
    daily = rep(.3, etf_ncol),
    weekly = rep(.2, etf_ncol),
    monthly = rep(.1, etf_ncol)
  )
  fit_test_bvhar_vhar <- bvhar_minnesota(
    y = etf_vix[1:50, 1:3],
    num_iter = 10,
    num_burn = 0,
    bayes_spec = vhar_spec
  )
  
  expect_s3_class(vhar_spec, "bvharspec")
  expect_s3_class(fit_test_bvhar_vhar, "bvharmn")
  
  expect_equal(
    nrow(fit_test_bvhar_vhar$coef), 
    ncol(etf_vix[1:50, 1:3]) * 3 + 1
  )
  expect_equal(
    ncol(fit_test_bvhar_vhar$coef),
    ncol(etf_vix[1:50, 1:3])
  )
  
})
#> Test passed 🌈