# bvar_sv()-------------------------
test_that("Members", {
  fit_test <- bvar_sv(
    etf_vix[1:50, 1:3],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvar(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "bvarsv")
  
  fit_test <- bvar_sv(
    etf_vix[1:50, 1:3],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda_record", "tau_record", "kappa_record") %in% names(fit_test)))
  
  fit_test <- bvar_sv(
    etf_vix[1:50, 1:3],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ssvs(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma_record" %in% names(fit_test))
})

test_that("Multi chain", {
  iter_test <- 5
  chain_test <- 2
  fit_test <- bvar_sv(
    etf_vix[1:50, 1:3],
    p = 1,
    num_chains = chain_test,
    num_iter = iter_test,
    num_burn = 0,
    thinning = 1,
    include_mean = FALSE
  )
  expect_equal(
    nrow(fit_test$param),
    iter_test * chain_test
  )
})
#> Test passed 🌈