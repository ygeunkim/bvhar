test_that("Test for hyperparameter functions", {
  bvar_spec <- set_bvar(
    sigma = rep(0, 3),
    lambda = 0,
    delta = rep(.2, 3)
  )
  
  expect_s3_class(bvar_spec, "bvharspec")
  
  
})
