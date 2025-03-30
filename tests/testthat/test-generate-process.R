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
