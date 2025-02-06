test_that("Construct coefficient names", {
  name_lag <- concatenate_colnames(c("x", "y"), 1:2, FALSE)
  expect_equal(name_lag, c("x_1", "y_1", "x_2", "y_2"))
})
