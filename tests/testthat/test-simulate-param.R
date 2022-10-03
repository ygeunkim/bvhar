# BVAR simlation---------------------
test_that("BVAR simulation", {
  test_lag <- 2
  num_col <- 3
  bvar_spec <- set_bvar(
    sigma = rep(1, num_col),
    lambda = .1,
    delta = rep(.1, num_col),
    eps = 1e-04
  )
  set.seed(1)
  mat_bvar_full <- sim_mncoef(
    p = test_lag,
    bayes_spec = bvar_spec,
    full = TRUE
  )
  mat_bvar_notfull <- sim_mncoef(
    p = test_lag,
    bayes_spec = bvar_spec,
    full = FALSE
  )
  
  # name of list------------
  expect_named(
    mat_bvar_full,
    c("coefficients", "covmat")
  )
  expect_named(
    mat_bvar_notfull,
    c("coefficients", "covmat")
  )
  
  # size--------------------
  expect_equal(
    nrow(mat_bvar_full$coefficients), 
    num_col * test_lag
  )
  expect_equal(
    ncol(mat_bvar_full$coefficients), 
    num_col
  )
  expect_equal(
    nrow(mat_bvar_full$covmat), 
    num_col
  )
  expect_equal(
    ncol(mat_bvar_full$covmat), 
    num_col
  )
  
  # when full = FALSE-------
  expect_identical(
    mat_bvar_notfull$covmat,
    diag(bvar_spec$sigma)
  )
})
#> Test passed 🌈

# BVHAR simlation--------------------
test_that("BVHAR simulation", {
  num_col <- 3
  bvhar_spec <- set_bvhar(
    sigma = rep(1, num_col),
    lambda = .1,
    delta = rep(.1, num_col),
    eps = 1e-04
  )
  set.seed(1)
  mat_bvhar_full <- sim_mnvhar_coef(
    bayes_spec = bvhar_spec,
    full = TRUE
  )
  mat_bvhar_notfull <- sim_mnvhar_coef(
    bayes_spec = bvhar_spec,
    full = FALSE
  )
  
  # name of list------------
  expect_named(
    mat_bvhar_full,
    c("coefficients", "covmat")
  )
  expect_named(
    mat_bvhar_notfull,
    c("coefficients", "covmat")
  )
  
  # size--------------------
  expect_equal(
    nrow(mat_bvhar_full$coefficients), 
    num_col * 3
  )
  expect_equal(
    ncol(mat_bvhar_full$coefficients), 
    num_col
  )
  expect_equal(
    nrow(mat_bvhar_full$covmat), 
    num_col
  )
  expect_equal(
    ncol(mat_bvhar_full$covmat), 
    num_col
  )
  
  # when full = FALSE-------
  expect_identical(
    mat_bvhar_notfull$covmat,
    diag(bvhar_spec$sigma)
  )
})
#> Test passed 🌈
