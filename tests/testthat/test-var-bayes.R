# var_bayes()-------------------------
test_that("VAR-Minn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvar(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VAR-HS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
})

test_that("VAR-SSVS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ssvs(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("VAR-Hierminn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvar(lambda = set_lambda()),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VAR-NG-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ng(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VAR-DL-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_dl(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VAR-GDP-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_gdp(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VAR-Minn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvar(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("VAR-HS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
})

test_that("VAR-SSVS-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ssvs(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("VAR-Hierminn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_bvar(lambda = set_lambda()),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("VAR-NG-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_ng(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VAR-DL-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_dl(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VAR-DL-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_gdp(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Multi chain", {
  skip_on_cran()
  
  set.seed(1)
  iter_test <- 5
  chain_test <- 2
  fit_test <- var_bayes(
    etf_vix[1:50, 1:2],
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