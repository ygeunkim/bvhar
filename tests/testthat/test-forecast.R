# VAR----------------------------------
test_that("Test for varlse and vharlse forecast", {
  skip_on_cran()
  
  num_col <- 3
  etf_train <- etf_vix[1:50, 1:num_col]
  etf_x <- etf_vix[1:50, (num_col + 1):(num_col + 2)]
  fit_var <- var_lm(etf_train, 2)
  fit_vhar <- vhar_lm(etf_train)
  fit_varx <- var_lm(etf_train, 2, exogen = etf_x)
  fit_vharx <- vhar_lm(etf_train, exogen = etf_x)
  
  num_forecast <- 2
  pred_var <- predict(fit_var, num_forecast)
  pred_varx <- predict(fit_varx, num_forecast, newxreg = etf_vix[51:(50 + num_forecast), (num_col + 1):(num_col + 2)])
  pred_vhar <- predict(fit_vhar, num_forecast)
  pred_vharx <- predict(fit_vharx, num_forecast, newxreg = etf_vix[51:(50 + num_forecast), (num_col + 1):(num_col + 2)])
  
  expect_s3_class(pred_var, "predbvhar")
  expect_s3_class(pred_vhar, "predbvhar")

  expect_equal(
    nrow(pred_var$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_var$forecast),
    num_col
  )
  expect_equal(
    nrow(pred_vhar$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_vhar$forecast),
    num_col
  )
  expect_error(
    predict(fit_varx, num_forecast),
    "'newxreg' should be supplied when using VARX model."
  )
  expect_error(
    predict(fit_vharx, num_forecast),
    "'newxreg' should be supplied when using VHARX model."
  )
})

help_var_bayes_pred <- function(bayes_spec, cov_spec, sparse) {
  num_forecast <- 3
  etf_train <- etf_vix[1:50, 1:2]
  etf_exog_train <- etf_vix[1:50, 3:4]
  etf_exog_new <- etf_vix[51:(50 + num_forecast), 3:4]

  set.seed(1)
  fit_test <- var_bayes(
    etf_train,
    p = 1,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bfavar <- var_bayes(
    etf_train,
    p = 1,
    factor_spec = set_factor(size_factor = 1),
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    loading_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bvarx <- var_bayes(
    etf_train,
    p = 1,
    exogen = etf_exog_train,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bfavarx <- var_bayes(
    etf_train,
    p = 1,
    exogen = etf_exog_train,
    factor_spec = set_factor(size_factor = 1),
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  predict(fit_test, num_forecast, sparse = sparse)
  set.seed(1)
  predict(fit_bfavar, num_forecast, sparse = sparse)
  set.seed(1)
  predict(fit_bvarx, num_forecast, newxreg = etf_exog_new, sparse = sparse)
  set.seed(1)
  predict(fit_bfavarx, num_forecast, newxreg = etf_exog_new, sparse = sparse)
  set.seed(1)
  predict(fit_test, sparse = sparse)
  set.seed(1)
  predict(fit_bfavar, sparse = sparse)
  set.seed(1)
  predict(fit_bvarx, sparse = sparse)
  set.seed(1)
  predict(fit_bfavarx, newxreg = etf_exog_new, sparse = sparse)
}

test_that("Forecast - VAR-HS-LDLT", {
  skip_on_cran()

  test_pred_dense <- help_var_bayes_pred(set_horseshoe(), set_ldlt(), FALSE)
  test_pred_sparse <- help_var_bayes_pred(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_pred_dense, "predbvhar")

  expect_s3_class(test_pred_sparse, "predbvhar")
})

test_that("Forecast - VAR-HS-SV", {
  skip_on_cran()

  expect_no_error({
    test_pred_dense <- help_var_bayes_pred(set_horseshoe(), set_sv(), FALSE)
    test_pred_sparse <- help_var_bayes_pred(set_horseshoe(), set_sv(), TRUE)
  })
})

help_vhar_bayes_pred <- function(bayes_spec, cov_spec, sparse) {
  num_forecast <- 3
  etf_train <- etf_vix[1:50, 1:2]
  etf_exog_train <- etf_vix[1:50, 3:4]
  etf_exog_new <- etf_vix[51:(50 + num_forecast), 3:4]

  set.seed(1)
  fit_test <- vhar_bayes(
    etf_train,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bfavhar <- vhar_bayes(
    etf_train,
    factor_spec = set_factor(size_factor = 1),
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    loading_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bvharx <- vhar_bayes(
    etf_train,
    exogen = etf_exog_train,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  fit_bfavharx <- vhar_bayes(
    etf_train,
    exogen = etf_exog_train,
    factor_spec = set_factor(size_factor = 1),
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )

  set.seed(1)
  predict(fit_test, num_forecast, sparse = sparse)
  set.seed(1)
  predict(fit_bfavhar, num_forecast, sparse = sparse)
  set.seed(1)
  predict(fit_bvharx, num_forecast, newxreg = etf_exog_new, sparse = sparse)
  set.seed(1)
  predict(fit_bfavharx, num_forecast, newxreg = etf_exog_new, sparse = sparse)
  set.seed(1)
  predict(fit_test, sparse = sparse)
  predict(fit_bfavhar, sparse = sparse)
  set.seed(1)
  predict(fit_bvharx, sparse = sparse)
  predict(fit_bfavharx, sparse = sparse)
}

test_that("Forecast - VHAR-Minn-LDLT", {
  skip_on_cran()

  test_pred_dense <- help_vhar_bayes_pred(set_bvhar(), set_ldlt(), FALSE)
  test_pred_sparse <- help_vhar_bayes_pred(set_bvhar(), set_ldlt(), TRUE)

  expect_s3_class(test_pred_dense, "predbvhar")

  expect_s3_class(test_pred_sparse, "predbvhar")
})

test_that("Forecast - VHAR-Minn-SV", {
  skip_on_cran()

  expect_no_error({
    test_pred_dense <- help_vhar_bayes_pred(set_bvhar(), set_sv(), FALSE)
    test_pred_sparse <- help_vhar_bayes_pred(set_bvhar(), set_sv(), TRUE)
  })
})
#> Test passed 🌈