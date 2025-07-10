test_that("MN parameterization", {
  set.seed(1)
  x <- sim_matgaussian(matrix(1:20, nrow = 4), diag(rep(3, 4)), diag(5), FALSE)
  set.seed(1)
  y <- sim_matgaussian(matrix(1:20, nrow = 4), diag(rep(1 / 3, 4)), diag(5), TRUE)
  expect_equal(x, y)
})

test_that("MNIW parameterization", {
  set.seed(1)
  x <- sim_mniw(1, matrix(1:20, nrow = 4), diag(rep(3, 4)), diag(5), 7, FALSE)$mn[[1]]
  set.seed(1)
  y <- sim_mniw(1, matrix(1:20, nrow = 4), diag(rep(1 / 3, 4)), diag(5), 7, TRUE)$mn[[1]]
  expect_equal(x, y)
})

test_that("DGP - VAR", {
  skip_on_cran()
  fit_test_var <- var_lm(y = etf_vix[1:30, 1:3], p = 2)
  
  expect_no_error({
    set.seed(1)
    x_normal_eigen <- sim_var(
      num_sim = 5,
      num_burn = 3,
      var_coef = fit_test_var$coefficients,
      var_lag = fit_test_var$p,
      sig_error = fit_test_var$covmat,
      method = "eigen",
      process = "gaussian"
    )
  })
  expect_no_error({
    set.seed(1)
    x_normal_chol <- sim_var(
      num_sim = 5,
      num_burn = 3,
      var_coef = fit_test_var$coefficients,
      var_lag = fit_test_var$p,
      sig_error = fit_test_var$covmat,
      method = "chol",
      process = "gaussian"
    )
  })
  expect_no_error({
    set.seed(1)
    x_student_eigen <- sim_var(
      num_sim = 5,
      num_burn = 3,
      var_coef = fit_test_var$coefficients,
      var_lag = fit_test_var$p,
      sig_error = fit_test_var$covmat,
      method = "eigen",
      process = "student",
      t_param = 5
    )
  })
  expect_no_error({
    set.seed(1)
    x_student_chol <- sim_var(
      num_sim = 5,
      num_burn = 3,
      var_coef = fit_test_var$coefficients,
      var_lag = fit_test_var$p,
      sig_error = fit_test_var$covmat,
      method = "chol",
      process = "student",
      t_param = 5
    )
  })
})

test_that("DGP - VHAR", {
  skip_on_cran()
  fit_test <- vhar_lm(y = etf_vix[1:50, 1:3])

  expect_no_error({
    set.seed(1)
    x_normal_eigen <- sim_vhar(
      num_sim = 5,
      num_burn = 3,
      vhar_coef = fit_test$coefficients,
      sig_error = fit_test$covmat,
      method = "eigen",
      process = "gaussian"
    )
  })
  expect_no_error({
    set.seed(1)
    x_normal_chol <- sim_vhar(
      num_sim = 5,
      num_burn = 3,
      vhar_coef = fit_test$coefficients,
      sig_error = fit_test$covmat,
      method = "chol",
      process = "gaussian"
    )
  })
  expect_no_error({
    set.seed(1)
    x_student_eigen <- sim_vhar(
      num_sim = 5,
      num_burn = 3,
      vhar_coef = fit_test$coefficients,
      sig_error = fit_test$covmat,
      method = "eigen",
      process = "student",
      t_param = 5
    )
  })
  expect_no_error({
    set.seed(1)
    x_student_chol <- sim_vhar(
      num_sim = 5,
      num_burn = 3,
      vhar_coef = fit_test$coefficients,
      sig_error = fit_test$covmat,
      method = "chol",
      process = "student",
      t_param = 5
    )
  })
})
