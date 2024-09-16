# Model evaluation of VAR----------------
test_that("VAR evaluation", {
  skip_on_cran()
  
  test_lag <- 3
  fit_test_var <- var_lm(etf_vix, test_lag)
  expect_s3_class(fit_test_var, "varlse")
  
  expect_s3_class(logLik(fit_test_var), "logLik")
  
})
#> Test passed 🌈

# Model evaluation of VAR----------------
test_that("VHAR evaluation", {
  skip_on_cran()
  
  fit_test_vhar <- vhar_lm(etf_vix)
  expect_s3_class(logLik(fit_test_vhar), "logLik")
  
})
#> Test passed 🌈

# Model evaluation of VAR----------------
test_that("VAR evaluation", {
  skip_on_cran()
  
  fit_test_vhar <- vhar_lm(etf_vix)
  expect_s3_class(fit_test_vhar, "vharlse")
  
  expect_s3_class(logLik(fit_test_vhar), "logLik")
  
})
#> Test passed 🌈

# Model evaluation of BVAR----------------
test_that("BVAR evaluation", {
  skip_on_cran()
  
  test_lag <- 3
  etf_ncol <- ncol(etf_vix)
  bvar_spec <- set_bvar(
    sigma = apply(etf_vix, 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvar <- bvar_minnesota(etf_vix, test_lag, bayes_spec = bvar_spec)
  expect_s3_class(fit_test_bvar, "bvarmn")
  
  expect_s3_class(logLik(fit_test_bvar), "logLik")
  
})
#> Test passed 🌈

# Model evaluation of BVHAR-S----------------
test_that("BVHAR-S evaluation", {
  skip_on_cran()
  
  har <- c(5, 22)
  etf_ncol <- ncol(etf_vix)
  vhar_spec <- set_bvhar(
    sigma = apply(etf_vix, 2, sd),
    lambda = .2,
    delta = rep(.1, etf_ncol)
  )
  fit_test_bvhar_s <- bvhar_minnesota(etf_vix, har = har, bayes_spec = vhar_spec)
  expect_s3_class(fit_test_bvhar_s, "bvharmn")
  
  expect_s3_class(logLik(fit_test_bvhar_s), "logLik")
  
})
#> Test passed 🌈

# Model evaluation of BVHAR-L----------------
test_that("BVHAR-L evaluation", {
  skip_on_cran()
  
  har <- c(5, 22)
  etf_ncol <- ncol(etf_vix)
  vhar_spec <- set_weight_bvhar(
    sigma = apply(etf_vix, 2, sd),
    lambda = .2,
    daily = rep(.3, etf_ncol),
    weekly = rep(.2, etf_ncol),
    monthly = rep(.1, etf_ncol)
  )
  
  fit_test_bvhar_s <- bvhar_minnesota(etf_vix, har = har, bayes_spec = vhar_spec)
  expect_s3_class(fit_test_bvhar_s, "bvharmn")
  
  expect_s3_class(logLik(fit_test_bvhar_s), "logLik")
  
})
#> Test passed 🌈
