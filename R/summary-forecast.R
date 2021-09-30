#' Split a Time Series Dataset into Train-Test Set
#' 
#' Split a given time series dataset into train and test set for evaluation.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param n.ahead step to evaluate
#' 
#' @seealso 
#' \code{\link[rsample:initial_time_split]{rsample::initial_time_split}}, \code{\link[rsample:training]{rsample::training}}, and \code{\link[rsample:testing]{rsample::testing}} process
#' provides tidyverse solution.
#' 
#' @importFrom stats setNames
#' @export
divide_ts <- function(y, n.ahead) {
  num_ts <- nrow(y)
  fac_train <- rep(1, num_ts - n.ahead)
  fac_test <- rep(2, n.ahead)
  y %>% 
    split.data.frame(
      factor(c(fac_train, fac_test))
    ) %>% 
    setNames(c("train", "test"))
}

#' Evaluate the Model Based on MSPE (Mean Squared Prediction Error)
#' @param x \code{predbvhar} object
#' @param y test data to be compared
#' @param ... not used
#' 
#' @export
mse <- function(x, y, ...) {
  UseMethod("mse", x)
}

#' Compute MSE
#' 
#' @param x \code{predbvhar} object
#' @param y test data to be compared. should be the same format with the train data and predict$forecast.
#' @param ... not used
#' 
#' @export
mse.predbvhar <- function(x, y, ...) {
  apply(y - x$forecast, 2, function(x) mean(x^2))
}

#' Evaluate the Model Based on MAPE (Mean Absolute Percentage Error)
#' @param x \code{predbvhar} object
#' @param y test data to be compared
#' @param ... not used
#' 
#' @export
mape <- function(x, y, ...) {
  UseMethod("mape", x)
}

#' Compute MAPE
#' 
#' @param x \code{predbvhar} object
#' @param y test data to be compared. should be the same format with the train data and predict$forecast.
#' @param ... not used
#' 
#' @export
mape.predbvhar <- function(x, y, ...) {
  apply((y - x$forecast) / y, 2, function(x) mean(abs(x)))
}
