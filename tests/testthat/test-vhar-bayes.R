help_vhar_bayes <- function(coef_spec, contem_spec = coef_spec, cov_spec, exogen_spec = NULL, loading_spec = NULL, include_mean = FALSE) {
  vix_endog <- etf_vix[1:50, 1:2]
  vix_exog <- NULL
  if (!is.null(exogen_spec)) {
    vix_exog <- etf_vix[1:50, 3:4]
  }
  factor_spec <- set_factor(size_factor = 0, factor_lag = 0)
  if (!is.null(loading_spec)) {
    factor_spec <- set_factor(size_factor = 1)
  }
  set.seed(1)
  vhar_bayes(
    y = vix_endog,
    har = c(5, 22),
    exogen = vix_exog,
    s = 0,
    factor_spec = factor_spec,
    num_iter = 5,
    num_burn = 0,
    coef_spec = coef_spec,
    contem_spec = contem_spec,
    exogen_spec = exogen_spec,
    loading_spec = loading_spec,
    cov_spec = cov_spec,
    include_mean = include_mean
  )
}

# vhar_bayes()-------------------------
test_that("VHAR-Minn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(),
    contem_spec = set_bvhar(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VHAR-HS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_horseshoe(),
    contem_spec = set_horseshoe(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
  expect_no_error({
    fit_test <- help_vhar_bayes(
      coef_spec = set_horseshoe(),
      contem_spec = set_horseshoe(),
      cov_spec = set_ldlt(),
      exogen_spec = NULL,
      loading_spec = set_horseshoe(),
      include_mean = FALSE
    )
  })
})

test_that("VHAR-SSVS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ssvs(),
    contem_spec = set_ssvs(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("VHAR-Hierminn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(lambda = set_lambda()),
    contem_spec = set_bvhar(lambda = set_lambda()),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VHAR-NG-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ng(),
    contem_spec = set_ng(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VHAR-DL-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_dl(),
    contem_spec = set_dl(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VHAR-GDP-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_gdp(),
    contem_spec = set_gdp(),
    cov_spec = set_ldlt(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VHARX-Minn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(),
    contem_spec = set_bvhar(),
    cov_spec = set_ldlt(),
    exogen_spec = set_bvhar(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VHARX-HS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_horseshoe(),
    contem_spec = set_horseshoe(),
    cov_spec = set_ldlt(),
    exogen_spec = set_horseshoe(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
  expect_no_error({
    fit_test <- help_vhar_bayes(
      coef_spec = set_horseshoe(),
      contem_spec = set_horseshoe(),
      cov_spec = set_ldlt(),
      exogen_spec = set_horseshoe(),
      loading_spec = set_horseshoe(),
      include_mean = TRUE
    )
  })
})

test_that("VHARX-SSVS-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ssvs(),
    contem_spec = set_ssvs(),
    cov_spec = set_ldlt(),
    exogen_spec = set_ssvs(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("VHARX-Hierminn-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(lambda = set_lambda()),
    contem_spec = set_bvhar(lambda = set_lambda()),
    cov_spec = set_ldlt(),
    exogen_spec = set_bvhar(lambda = set_lambda()),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ldltmod")
})

test_that("VHARX-NG-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ng(),
    contem_spec = set_ng(),
    cov_spec = set_ldlt(),
    exogen_spec = set_ng(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VHARX-DL-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_dl(),
    contem_spec = set_dl(),
    cov_spec = set_ldlt(),
    exogen_spec = set_dl(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("VHARX-GDP-LDLT", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_gdp(),
    contem_spec = set_gdp(),
    cov_spec = set_ldlt(),
    exogen_spec = set_gdp(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHAR-Minn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(),
    contem_spec = set_bvhar(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("Members - VHAR-HS-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_horseshoe(),
    contem_spec = set_horseshoe(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
})

test_that("Members - VHAR-SSVS-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ssvs(),
    contem_spec = set_ssvs(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("Members - VHAR-Hierminn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(lambda = set_lambda()),
    contem_spec = set_bvhar(lambda = set_lambda()),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("Members - VHAR-NG-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ng(),
    contem_spec = set_ng(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHAR-DL-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_dl(),
    contem_spec = set_dl(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHAR-GDP-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_gdp(),
    contem_spec = set_gdp(),
    cov_spec = set_sv(),
    exogen_spec = NULL,
    include_mean = FALSE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHARX-Minn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(),
    contem_spec = set_bvhar(),
    cov_spec = set_sv(),
    exogen_spec = set_bvhar(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("Members - VHARX-HS-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_horseshoe(),
    contem_spec = set_horseshoe(),
    cov_spec = set_sv(),
    exogen_spec = set_horseshoe(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "hsmod")
  expect_true(all(c("lambda", "tau", "kappa") %in% fit_test$param_names))
})

test_that("Members - VHARX-SSVS-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ssvs(),
    contem_spec = set_ssvs(),
    cov_spec = set_sv(),
    exogen_spec = set_ssvs(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ssvsmod")
  expect_true("gamma" %in% fit_test$param_names)
})

test_that("Members - VHARX-Hierminn-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_bvhar(lambda = set_lambda()),
    contem_spec = set_bvhar(lambda = set_lambda()),
    cov_spec = set_sv(),
    exogen_spec = set_bvhar(lambda = set_lambda()),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "svmod")
})

test_that("Members - VHARX-NG-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_ng(),
    contem_spec = set_ng(),
    cov_spec = set_sv(),
    exogen_spec = set_ng(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "ngmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHARX-DL-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_dl(),
    contem_spec = set_dl(),
    cov_spec = set_sv(),
    exogen_spec = set_dl(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "dlmod")
  expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Members - VHARX-GDP-SV", {
  skip_on_cran()

  set.seed(1)
  fit_test <- help_vhar_bayes(
    coef_spec = set_gdp(),
    contem_spec = set_gdp(),
    cov_spec = set_sv(),
    exogen_spec = set_gdp(),
    include_mean = TRUE
  )
  expect_s3_class(fit_test, "gdpmod")
  # expect_true(all(c("lambda", "tau") %in% fit_test$param_names))
})

test_that("Multi chain", {
  skip_on_cran()
  
  set.seed(1)
  iter_test <- 5
  chain_test <- 2
  fit_test <- vhar_bayes(
    etf_vix[1:50, 1:2],
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