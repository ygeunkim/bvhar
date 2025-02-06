# spillover() and dynamic_spillover()--------------
test_that("VAR-Spillover", {
  skip_on_ci()
  skip_on_cran()

  num_row <- 30
  win_size <- 29
  num_col <- 2
  fit_test <- var_lm(etf_vix[seq_len(num_row), seq_len(num_col)], p = 1)
  sp_test <- spillover(fit_test, n_ahead = 10)
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10, window = win_size)
  expect_s3_class(sp_test, "bvharspillover")
  expect_equal(sp_test$process, fit_test$process)
  expect_equal(dim(sp_test$connect), c(num_col + 1, num_col + 1))
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(length(sp_dynamic_test$tot), num_row - win_size + 1)
})

test_that("VHAR-Spillover", {
  skip_on_ci()
  skip_on_cran()

  num_row <- 30
  win_size <- 29
  num_col <- 2
  fit_test <- vhar_lm(etf_vix[seq_len(num_row), seq_len(num_col)])
  sp_test <- spillover(fit_test, n_ahead = 10)
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10, window = win_size)
  expect_s3_class(sp_test, "bvharspillover")
  expect_equal(sp_test$process, fit_test$process)
  expect_equal(dim(sp_test$connect), c(num_col + 1, num_col + 1))
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(length(sp_dynamic_test$tot), num_row - win_size + 1)
})

test_that("VAR-LDLT-Spillover", {
  skip_on_ci()
  skip_on_cran()

  num_row <- 30
  win_size <- 29
  num_col <- 2
  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[seq_len(num_row), seq_len(num_col)],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  sp_test <- spillover(fit_test, n_ahead = 10)
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10, window = win_size)
  expect_s3_class(sp_test, "bvharspillover")
  expect_equal(sp_test$process, fit_test$process)
  expect_equal(dim(sp_test$connect[[1]]), c(num_col, num_col))
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(nrow(sp_dynamic_test$tot), num_row - win_size + 1)
})

test_that("VHAR-LDLT-Spillover", {
  skip_on_ci()
  skip_on_cran()

  num_row <- 30
  win_size <- 29
  num_col <- 2
  set.seed(1)
  fit_test <- vhar_bayes(
    etf_vix[seq_len(num_row), seq_len(num_col)],
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_ldlt(),
    include_mean = FALSE
  )
  sp_test <- spillover(fit_test, n_ahead = 10)
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10, window = win_size)
  expect_s3_class(sp_test, "bvharspillover")
  expect_equal(sp_test$process, fit_test$process)
  expect_equal(dim(sp_test$connect[[1]]), c(num_col, num_col))
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(nrow(sp_dynamic_test$tot), num_row - win_size + 1)
})

test_that("VAR-SV-Spillover", {
  skip_on_ci()
  skip_on_cran()

  num_row <- 30
  num_col <- 2
  var_lag <- 1
  set.seed(1)
  fit_test <- var_bayes(
    etf_vix[seq_len(num_row), seq_len(num_col)],
    p = 1,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10)
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(nrow(sp_dynamic_test$tot), num_row - var_lag)
})

test_that("VHAR-SV-Spillover", {
  skip_on_ci()
  skip_on_cran()
  
  num_row <- 30
  num_col <- 2
  har_lag <- c(5, 22)
  set.seed(1)
  fit_test <- vhar_bayes(
    etf_vix[seq_len(num_row), seq_len(num_col)],
    har = har_lag,
    num_iter = 5,
    num_burn = 0,
    bayes_spec = set_horseshoe(),
    cov_spec = set_sv(),
    include_mean = FALSE
  )
  sp_dynamic_test <- dynamic_spillover(fit_test, n_ahead = 10)
  expect_s3_class(sp_dynamic_test, "bvhardynsp")
  expect_equal(nrow(sp_dynamic_test$tot), num_row - har_lag[2])
})
#> Test passed 🌈