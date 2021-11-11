#' Split a Time Series Dataset into Train-Test Set
#' 
#' Split a given time series dataset into train and test set for evaluation.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param n_ahead step to evaluate
#' 
#' @seealso 
#' [rsample::initial_time_split()], [rsample::training()], and [rsample::testing()] process provides tidyverse solution.
#' 
#' @importFrom stats setNames
#' @export
divide_ts <- function(y, n_ahead) {
  num_ts <- nrow(y)
  fac_train <- rep(1, num_ts - n_ahead)
  fac_test <- rep(2, n_ahead)
  y %>% 
    split.data.frame(
      factor(c(fac_train, fac_test))
    ) %>% 
    setNames(c("train", "test"))
}

#' Out-of-sample Forecasting based on Rolling Window
#' 
#' This function forecast for out-of-sample.
#' 
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared. Use [divide_ts()] if you don't have separate evaluation dataset.
#' @details 
#' Rolling windows forecasting fixes window size.
#' It moves the window ahead and forecast h-ahead in `y_test` set.
#' 
#' @order 1
#' @export
forecast_roll <- function(object, n_ahead, y_test) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  num_test <- nrow(y_test)
  n_iter <- num_test - n_ahead + 1
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  res_mat <- switch(
    model_type,
    "varlse" = {
      roll_var(y, object$p, include_mean, n_ahead, n_iter)
    },
    "vharlse" = {
      roll_vhar(y, include_mean, n_ahead, n_iter)
    },
    "bvarmn" = {
      roll_bvar(y, object$p, object$spec, include_mean, n_ahead, n_iter)
    },
    "bvarflat" = {
      roll_bvarflat(y, object$p, object$spec, include_mean, n_ahead, n_iter)
    },
    "bvharmn" = {
      roll_bvhar(y, object$spec, include_mean, n_ahead, n_iter)
    }
  )
  colnames(res_mat) <- name_var
  res <- list(
    forecast = res_mat,
    evaluation = y_test[n_ahead:num_test,],
    y = y
  )
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' Evaluate the Model Based on MSE (Mean Square Error)
#' 
#' This function computes MSE given prediction result versus evaluation set.
#' 
#' @param x `predbvhar` or `bvharcv` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' 
#' @export
mse <- function(x, y, ...) {
  UseMethod("mse", x)
}

#' @rdname mse
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}. Then
#' \deqn{MSE = mean(e_t^2)}
#' MSE is the most used accuracy measure.
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. doi:[10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
#' 
#' @export
mse.predbvhar <- function(x, y, ...) {
  (y - x$forecast)^2 %>% 
    colMeans()
}

#' @rdname mse
#' 
#' @param x `bvharcv` object
#' @param ... not used
#' 
#' @export
mse.bvharcv <- function(x, ...) {
  (x$evaluation - x$forecast)^2 %>% 
    colMeans()
}

#' Evaluate the Model Based on MAE (Mean Absolute Error)
#' 
#' This function computes MAE given prediction result versus evaluation set.
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' 
#' @export
mae <- function(x, y, ...) {
  UseMethod("mae", x)
}

#' @rdname mae
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' MAE is defined by
#' 
#' \deqn{MSE = mean(\lvert e_t \rvert)}
#' 
#' Some researchers prefer MAE to MSE because it is less sensitive to outliers.
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. doi:[10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
#' 
#' @export
mae.predbvhar <- function(x, y, ...) {
  apply(
    y - x$forecast, 
    2, 
    function(e_t) {
      mean(abs(e_t))
    }
  )
}

#' @rdname mae
#' 
#' @param x `bvharcv` object
#' @param ... not used
#' 
#' @export
mae.bvharcv <- function(x, ...) {
  apply(
    x$evaluation - x$forecast,
    2,
    function(e_t) {
      mean(abs(e_t))
    }
  )
}

#' Evaluate the Model Based on MAPE (Mean Absolute Percentage Error)
#' 
#' This function computes MAPE given prediction result versus evaluation set.
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' 
#' @export
mape <- function(x, y, ...) {
  UseMethod("mape", x)
}

#' @rdname mape
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' Percentage error is defined by \eqn{p_t = 100 e_t / Y_t} (100 can be omitted since comparison is the focus).
#' 
#' \deqn{MAPE = mean(\lvert p_t \rvert)}
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. doi:[10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
#' 
#' @export
mape.predbvhar <- function(x, y, ...) {
  apply(
    100 * (y - x$forecast) / y, 
    2, 
    function(p_t) {
      mean(abs(p_t))
    }
  )
}

#' @rdname mape
#' 
#' @param x `bvharcv` object
#' @param ... not used
#' 
#' @export
mape.bvharcv <- function(x, ...) {
  y_test <- x$evaluation
  apply(
    100 * (y_test - x$forecast) / y_test,
    2,
    function(p_t) {
      mean(abs(p_t))
    }
  )
}

#' Evaluate the Model Based on MASE (Mean Absolute Scaled Error)
#' 
#' This function computes MASE given prediction result versus evaluation set.
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' 
#' @export
mase <- function(x, y, ...) {
  UseMethod("mase", x)
}

#' @rdname mase
#' 
#' @param x `predbvhar` object
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' Scaled error is defined by
#' \deqn{q_t = \frac{e_t}{\sum_{i = 2}^{n} \lvert Y_i - Y_{i - 1} \rvert / (n - 1)}}
#' so that the error can be free of the data scale.
#' Then
#' 
#' \deqn{MASE = mean(\lvert q_t \rvert)}
#' 
#' Here, \eqn{Y_i} are the points in the sample, i.e. errors are scaled by the in-sample mean absolute error (\eqn{mean(\lvert e_t \rvert)}) from the naive random walk forecasting.
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. doi:[10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
#' 
#' @export
mase.predbvhar <- function(x, y, ...) {
  scaled_err <- 
    x$y %>% 
    diff() %>% 
    abs() %>% 
    colMeans()
  apply(
    100 * (y - x$forecast) / scaled_err, 
    2, 
    function(q_t) {
      mean(abs(q_t))
    }
  )
}

#' @rdname mase
#' 
#' @param x `bvharcv` object
#' @param ... not used
#' 
#' @export
mase.bvharcv <- function(x, ...) {
  scaled_err <- 
    x$y %>% 
    diff() %>% 
    abs() %>% 
    colMeans()
  y_test <- x$evaluation
  apply(
    100 * (y_test - x$forecast) / scaled_err, 
    2, 
    function(q_t) {
      mean(abs(q_t))
    }
  )
}

#' Evaluate the Model Based on MRAE (Mean Relative Absolute Error)
#' 
#' This function computes MRAE given prediction result versus evaluation set.
#' 
#' @param x `predbvhar` object to use
#' @param pred_bench `predbvhar` from benchmark model
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' 
#' @export
mrae <- function(x, pred_bench, y, ...) {
  UseMethod("mrae", x)
}

#' @rdname mrae
#' 
#' @param x `predbvhar` object to use
#' @param pred_bench `predbvhar` from benchmark model
#' @param y test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' MRAE implements benchmark model as scaling method.
#' Relative error is defined by
#' \deqn{r_t = \frac{e_t}{e_t^{\ast}}}
#' where \eqn{e_t^\ast} is the error from the benchmark method.
#' Then
#' 
#' \deqn{MRAE = mean(\lvert r_t \rvert)}
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688. doi:[10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
#' 
#' @export
mrae.predbvhar <- function(x, pred_bench, y, ...) {
  if (!is.predbvhar(pred_bench)) stop("'pred_bench' should be 'predbvhar' class.")
  apply(
    (y - x$forecast) / (y - pred_bench$forecast), 
    2, 
    function(r_t) {
      mean(abs(r_t))
    }
  )
}

#' Print Method for `bvharcv` object
#' @rdname forecast_roll
#' @param x `bvharcv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharcv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$forecast)
  invisible(x)
}

#' @rdname forecast_roll
#' @param x `bvharcv` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharcv <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharcv",
  knit_print.bvharcv,
  envir = asNamespace("knitr")
)
