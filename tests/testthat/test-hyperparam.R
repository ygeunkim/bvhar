# Bayesian model specification----------------
test_that("Test for hyperparameter functions", {
  skip_on_cran()
  
  bvar_spec <- set_bvar(
    sigma = rep(.1, 3),
    lambda = .1,
    delta = rep(.2, 3)
  )
  
  expect_s3_class(bvar_spec, "bvharspec")
  
  
})
#> Test passed 🌈