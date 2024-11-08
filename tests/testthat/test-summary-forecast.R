# Train-test---------------------------
test_that("Test for data-splitting", {
  skip_on_cran()
  
  test_data <- matrix(rnorm(10100), ncol = 10, byrow = TRUE)
  n_test <- 10
  n_train <- nrow(test_data) - n_test
  data_split <- divide_ts(test_data, n_test)
  
  expect_equal(
    nrow(data_split$train), 
    n_train
  )
  expect_equal(
    nrow(data_split$test), 
    n_test
  )
})

test_that("Rolling windows - OLS", {
  skip_on_cran()
  
  etf_split <- divide_ts(etf_vix[1:100, 1:2], 10)
  etf_train <- etf_split$train
  etf_test <- etf_split$test
  
  fit_var <- var_lm(etf_train, 2)
  var_roll <- forecast_roll(fit_var, 10, etf_test)
  expect_s3_class(var_roll, "predbvhar_roll")
  expect_s3_class(var_roll, "bvharcv")
  
})

test_that("Expanding windows - OLS", {
  skip_on_cran()
  
  etf_split <- divide_ts(etf_vix[1:100, 1:2], 10)
  etf_train <- etf_split$train
  etf_test <- etf_split$test
  
  fit_var <- var_lm(etf_train, 2)
  var_expand <- forecast_expand(fit_var, 10, etf_test)
  expect_s3_class(var_expand, "predbvhar_expand")
  expect_s3_class(var_expand, "bvharcv")
  
})

help_var_bayes_roll <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]
  etf_test <- etf_vix[51:53, 1:2]

  set.seed(1)
  fit_test <- var_bayes(
    etf_train,
    p = 1,
    num_iter = 3,
    num_burn = 0,
    bayes_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  forecast_roll(fit_test, 1, etf_test, stable = FALSE, sparse = sparse, lpl = TRUE)
}

help_vhar_bayes_roll <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]
  etf_test <- etf_vix[51:53, 1:2]

  set.seed(1)
  fit_test <- vhar_bayes(
    etf_train,
    num_iter = 3,
    num_burn = 0,
    bayes_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  forecast_roll(fit_test, 1, etf_test, stable = FALSE, sparse = sparse, lpl = TRUE)
}

help_var_bayes_expand <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]
  etf_test <- etf_vix[51:53, 1:2]

  set.seed(1)
  fit_test <- var_bayes(
    etf_train,
    p = 1,
    num_iter = 3,
    num_burn = 0,
    bayes_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  forecast_expand(fit_test, 1, etf_test, stable = FALSE, sparse = sparse, lpl = TRUE)
}

help_vhar_bayes_expand <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]
  etf_test <- etf_vix[51:53, 1:2]

  set.seed(1)
  fit_test <- vhar_bayes(
    etf_train,
    num_iter = 3,
    num_burn = 0,
    bayes_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  forecast_expand(fit_test, 1, etf_test, stable = FALSE, sparse = sparse, lpl = TRUE)
}

test_that("Rolling windows - VAR-Minn-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_bvar(), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_bvar(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-HS-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_horseshoe(), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-SSVS-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_ssvs(), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_ssvs(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-Hierminn-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_bvar(lambda = set_lambda()), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_bvar(lambda = set_lambda()), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-NG-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_ng(), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_ng(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-DL-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_dl(), set_ldlt(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_dl(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-Minn-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_bvhar(), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_bvhar(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-HS-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_horseshoe(), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-SSVS-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_ssvs(), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_ssvs(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-Hierminn-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_bvhar(lambda = set_lambda()), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_bvhar(lambda = set_lambda()), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-NG-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_ng(), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_ng(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-DL-LDLT", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_dl(), set_ldlt(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_dl(), set_ldlt(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-Minn-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_bvar(), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_bvar(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-HS-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_horseshoe(), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_horseshoe(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-SSVS-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_ssvs(), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_ssvs(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-Hierminn-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_bvar(lambda = set_lambda()), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_bvar(lambda = set_lambda()), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-NG-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_ng(), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_ng(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VAR-DL-SV", {
  skip_on_cran()

  test_roll_dense <- help_var_bayes_roll(set_dl(), set_sv(), FALSE)
  test_roll_sparse <- help_var_bayes_roll(set_dl(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-Minn-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_bvhar(), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_bvhar(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-HS-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_horseshoe(), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_horseshoe(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-SSVS-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_ssvs(), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_ssvs(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-Hierminn-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_bvhar(lambda = set_lambda()), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_bvhar(lambda = set_lambda()), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-NG-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_ng(), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_ng(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Rolling windows - VHAR-DL-SV", {
  skip_on_cran()

  test_roll_dense <- help_vhar_bayes_roll(set_dl(), set_sv(), FALSE)
  test_roll_sparse <- help_vhar_bayes_roll(set_dl(), set_sv(), TRUE)

  expect_s3_class(test_roll_dense, "predbvhar_roll")
  expect_s3_class(test_roll_dense, "bvharcv")

  expect_s3_class(test_roll_sparse, "predbvhar_roll")
  expect_s3_class(test_roll_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-Minn-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_bvar(), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_bvar(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-HS-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_horseshoe(), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-SSVS-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_ssvs(), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_ssvs(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-Hierminn-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_bvar(lambda = set_lambda()), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_bvar(lambda = set_lambda()), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-NG-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_ng(), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_ng(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-DL-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_dl(), set_ldlt(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_dl(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-Minn-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_bvhar(), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_bvhar(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-HS-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_horseshoe(), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-SSVS-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_ssvs(), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_ssvs(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-Hierminn-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_bvhar(lambda = set_lambda()), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_bvhar(lambda = set_lambda()), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-NG-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_ng(), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_ng(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-DL-LDLT", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_dl(), set_ldlt(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_dl(), set_ldlt(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-Minn-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_bvar(), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_bvar(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-HS-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_horseshoe(), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_horseshoe(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-SSVS-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_ssvs(), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_ssvs(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-Hierminn-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_bvar(lambda = set_lambda()), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_bvar(lambda = set_lambda()), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-NG-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_ng(), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_ng(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VAR-DL-SV", {
  skip_on_cran()

  test_expand_dense <- help_var_bayes_expand(set_dl(), set_sv(), FALSE)
  test_expand_sparse <- help_var_bayes_expand(set_dl(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-Minn-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_bvhar(), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_bvhar(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-HS-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_horseshoe(), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_horseshoe(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-SSVS-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_ssvs(), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_ssvs(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-Hierminn-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_bvhar(lambda = set_lambda()), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_bvhar(lambda = set_lambda()), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-NG-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_ng(), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_ng(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})

test_that("Expanding windows - VHAR-DL-SV", {
  skip_on_cran()

  test_expand_dense <- help_vhar_bayes_expand(set_dl(), set_sv(), FALSE)
  test_expand_sparse <- help_vhar_bayes_expand(set_dl(), set_sv(), TRUE)

  expect_s3_class(test_expand_dense, "predbvhar_expand")
  expect_s3_class(test_expand_dense, "bvharcv")

  expect_s3_class(test_expand_sparse, "predbvhar_expand")
  expect_s3_class(test_expand_sparse, "bvharcv")
})
#> Test passed 🌈