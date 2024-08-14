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

test_that("GIG generation - with mode shift", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- 3
  psi <- .2
  chi <- .5
  # beta <- sqrt(psi * chi)
  # if (lambda > 2 || beta > 3) {
  #   print("Mode shift")
  # } else if (lambda >= 1 - 9 * beta^2 / 4 || beta > .2) {
  #   print("without mode shift")
  # } else if (beta > 0) {
  #   print("non-concave")
  # }
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi)
  expect_equal(my_draw, orig_draw)
})

test_that("GIG generation - without mode shift", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- .8
  psi <- .3
  chi <- .5
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi)
  expect_equal(my_draw, orig_draw)
})

test_that("GIG generation - non-T_(-1/2)-concave part", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- .8
  psi <- .1
  chi <- .2
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi)
  expect_equal(my_draw, orig_draw)
})

test_that("GIG generation - Round-off handling (small chi)", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- 3
  psi <- .5
  chi <- 1e-50
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi)
  expect_equal(my_draw, orig_draw)
})

test_that("GIG generation - Round-off handling (small psi)", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- -3
  psi <- 1e-50
  chi <- .5
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi) # GIGrvg handling is different from bvhar
  expect_equal(my_draw, orig_draw)
})

test_that("GIG generation - Round-off handling (both small)", {
  skip_if_not_installed("GIGrvg")
  # skip_if(getRversion() < "4.0.0")
  lambda <- 3
  psi <- 1e-50
  chi <- 1e-50
  set.seed(1)
  my_draw <- sim_gig(10, lambda = lambda, psi = psi, chi = chi) # use rgamma(10, shape = abs(lambda), scale = 2 / psi)
  set.seed(1)
  orig_draw <- GIGrvg::rgig(10, lambda = lambda, chi = chi, psi = psi)
  expect_equal(my_draw, orig_draw)
})
