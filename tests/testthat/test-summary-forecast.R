# Train-test---------------------------
test_that("Test for data-splitting", {
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
#> Test passed 🌈