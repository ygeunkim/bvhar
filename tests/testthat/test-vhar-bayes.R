# vhar_bayes()-------------------------
test_that("Members", {
  fit_test <- vhar_bayes(
    etf_vix[1:50, 1:3],
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvhar(),
    include_mean = FALSE
  )
  # expect_s3_class(fit_test, "bvharsv")

  fit_test <- vhar_bayes(
    etf_vix[1:50, 1:3],
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))

  fit_test <- vhar_bayes(
    etf_vix[1:50, 1:3],
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ssvs(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("Multi chain", {
  iter_test <- 5
  chain_test <- 2
  fit_test <- vhar_bayes(
    etf_vix[1:50, 1:3],
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