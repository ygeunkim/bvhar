#' Split a Time Series Dataset into Train-Test Set
#' 
#' Split a given time series dataset into train and test set for evaluation.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param n_ahead step to evaluate
#' @return List of two datasets, train and test.
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
#' This function conducts rolling window forecasting.
#' 
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared. Use [divide_ts()] if you don't have separate evaluation dataset.
#' @param ... Additional arguments
#' @details 
#' Rolling windows forecasting fixes window size.
#' It moves the window ahead and forecast h-ahead in `y_test` set.
#' @return `predbvhar_roll` [class]
#' @seealso 
#' See [ts_forecasting_cv] for out-of-sample forecasting methods.
#' @references Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTEXTS.
#' @order 1
#' @export
forecast_roll <- function(object, n_ahead, y_test, ...) {
  UseMethod("forecast_roll", object)
}

#' @rdname forecast_roll
#' @export
forecast_roll.bvharmod <- function(object, n_ahead, y_test, ...) {
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
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_horizon <- nrow(y_test) - n_ahead + 1
  res_mat <- switch(model_type,
    "varlse" = {
      roll_var(y, object$p, include_mean, n_ahead, y_test)
    },
    "vharlse" = {
      roll_vhar(y, c(object$week, object$month), include_mean, n_ahead, y_test)
    },
    "bvarmn" = {
      roll_bvar(y, object$p, object$spec, include_mean, n_ahead, y_test)
    },
    "bvarflat" = {
      roll_bvarflat(y, object$p, object$spec, include_mean, n_ahead, y_test)
    },
    "bvharmn" = {
      roll_bvhar(y, c(object$week, object$month), object$spec, include_mean, n_ahead, y_test)
    }
  )
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' @rdname forecast_roll
#' @param innovation `r lifecycle::badge("experimental")` Include heteroskedastic covariance of innovation when forecasting. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @export
forecast_roll.svmod <- function(object, n_ahead, y_test, innovation = TRUE, use_fit = TRUE, num_thread = 1, ...) {
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
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning(sprintf("'num_thread' > MCMC chain will use not every thread. Specify as 'num_thread' <= 'object$chain' = %d.", num_chains))
  }
  if (num_horizon * num_chains %% num_thread != 0) {
    warning(sprintf("OpenMP cannot divide the iterations as integer. Use divisor of ('nrow(y_test) - n_ahead + 1') * 'num_thread' <= 'object$chain' = %d", num_horizon * num_chains))
  }
  chunk_size <- num_horizon * num_chains %/% num_thread # default setting of OpenMP schedule(static)
  if (chunk_size == 0) {
    chunk_size <- 1
  }
  if (num_horizon > num_chains && chunk_size > num_chains) {
    chunk_size <- min(
      num_chains,
      (num_horizon %/% num_thread) * num_chains
    )
    if (chunk_size == 0) {
      chunk_size <- 1
    }
  }
  fit_ls <- list()
  if (use_fit) {
    nm_record <- names(object)[grepl(pattern = "_record$", x = names(object))]
    fit_ls <-
      object[nm_record] %>%
      lapply(function(x) {
        as_draws_matrix(x) %>%
          split.data.frame(gl(num_chains, nrow(x) / num_chains))
      })
  }
  res_mat <- switch(model_type,
    "bvarsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      # param_init <- object$init
      if (is.bvharspec(object$spec)) {
        param_prior <- append(object$spec, list(p = object$p))
        prior_type <- 1
      } else if (is.ssvsinput(object$spec)) {
        param_prior <- object$spec
        prior_type <- 2
      } else {
        param_prior <- list()
        prior_type <- 3
      }
      roll_bvarsv(
        y, object$p, num_chains, object$iter, object$burn, object$thin,
        fit_ls,
        object$sv[3:6], param_prior, object$intercept, object$init, prior_type,
        grp_id, grp_mat,
        include_mean, n_ahead, y_test,
        innovation,
        sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
        sample.int(.Machine$integer.max, size = num_chains),
        num_thread, chunk_size
      )
    },
    "bvharsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      # param_init <- object$init
      if (is.bvharspec(object$spec)) {
        param_prior <- append(object$spec, list(p = 3))
        prior_type <- 1
      } else if (is.ssvsinput(object$spec)) {
        param_prior <- object$spec
        prior_type <- 2
      } else {
        param_prior <- list()
        prior_type <- 3
      }
      roll_bvharsv(
        y, object$week, object$month, num_chains, object$iter, object$burn, object$thin,
        fit_ls,
        object$sv[3:6], param_prior, object$intercept, object$init, prior_type,
        grp_id, grp_mat,
        include_mean, n_ahead, y_test,
        innovation,
        sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
        sample.int(.Machine$integer.max, size = num_chains),
        num_thread, chunk_size
      )
    }
  )
  num_draw <- nrow(object$a_record) # concatenate multiple chains
  res_mat <-
    res_mat %>%
    lapply(function(res) {
      unlist(res) %>%
        array(dim = c(1, object$m, num_draw)) %>%
        apply(c(1, 2), mean)
    }) %>%
    do.call(rbind, .)
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' Out-of-sample Forecasting based on Expanding Window
#'
#' This function conducts expanding window forecasting.
#'
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared. Use [divide_ts()] if you don't have separate evaluation dataset.
#' @param ... Additional arguments.
#' @details
#' Expanding windows forecasting fixes the starting period.
#' It moves the window ahead and forecast h-ahead in `y_test` set.
#' @return `predbvhar_expand` [class]
#' @seealso
#' See [ts_forecasting_cv] for out-of-sample forecasting methods.
#' @references Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTEXTS. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
#' @order 1
#' @export
forecast_expand <- function(object, n_ahead, y_test, ...) {
  UseMethod("forecast_expand", object)
}

#' @rdname forecast_expand
#' @export
forecast_expand.bvharmod <- function(object, n_ahead, y_test, ...) {
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
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  res_mat <- switch(
    model_type,
    "varlse" = {
      expand_var(y, object$p, include_mean, n_ahead, y_test)
    },
    "vharlse" = {
      expand_vhar(y, c(object$week, object$month), include_mean, n_ahead, y_test)
    },
    "bvarmn" = {
      expand_bvar(y, object$p, object$spec, include_mean, n_ahead, y_test)
    },
    "bvarflat" = {
      expand_bvarflat(y, object$p, object$spec, include_mean, n_ahead, y_test)
    },
    "bvharmn" = {
      expand_bvhar(y, c(object$week, object$month), object$spec, include_mean, n_ahead, y_test)
    }
  )
  num_horizon <- nrow(y_test) - n_ahead + 1
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_expand", "bvharcv")
  res
}

#' Evaluate the Model Based on MSE (Mean Square Error)
#' 
#' This function computes MSE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @return MSE vector corresponding to each variable.
#' @export
mse <- function(x, y, ...) {
  UseMethod("mse", x)
}

#' @rdname mse
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}. Then
#' \deqn{MSE = mean(e_t^2)}
#' MSE is the most used accuracy measure.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' @export
mse.predbvhar <- function(x, y, ...) {
  (y - x$forecast)^2 %>% 
    colMeans()
}

#' @rdname mse
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mse.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id,]
  (y_test - x$forecast)^2 %>% 
    colMeans()
}

#' Evaluate the Model Based on MAE (Mean Absolute Error)
#' 
#' This function computes MAE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MAE vector corressponding to each variable.
#' @export
mae <- function(x, y, ...) {
  UseMethod("mae", x)
}

#' @rdname mae
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' MAE is defined by
#' 
#' \deqn{MSE = mean(\lvert e_t \rvert)}
#' 
#' Some researchers prefer MAE to MSE because it is less sensitive to outliers.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
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
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mae.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id,]
  apply(
    y_test - x$forecast,
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
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MAPE vector corresponding to each variable.
#' @export
mape <- function(x, y, ...) {
  UseMethod("mape", x)
}

#' @rdname mape
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' Percentage error is defined by \eqn{p_t = 100 e_t / Y_t} (100 can be omitted since comparison is the focus).
#' 
#' \deqn{MAPE = mean(\lvert p_t \rvert)}
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
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
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mape.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id,]
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
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MASE vector corresponding to each variable.
#' @export
mase <- function(x, y, ...) {
  UseMethod("mase", x)
}

#' @rdname mase
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
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
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
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
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mase.bvharcv <- function(x, y, ...) {
  scaled_err <- 
    x$y %>% 
    diff() %>% 
    abs() %>% 
    colMeans()
  y_test <- y[x$eval_id,]
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
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MRAE vector corresponding to each variable.
#' @export
mrae <- function(x, pred_bench, y, ...) {
  UseMethod("mrae", x)
}

#' @rdname mrae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
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
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' @export
mrae.predbvhar <- function(x, pred_bench, y, ...) {
  if (!is.predbvhar(pred_bench)) {
    stop("'pred_bench' should be 'predbvhar' class.")
  }
  apply(
    (y - x$forecast) / (y - pred_bench$forecast), 
    2, 
    function(r_t) {
      mean(abs(r_t))
    }
  )
}

#' @rdname mrae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mrae.bvharcv <- function(x, pred_bench, y, ...) {
  if (!is.bvharcv(pred_bench)) {
    stop("'pred_bench' should be 'bvharcv' class.")
  }
  y_test <- y[x$eval_id,]
  apply(
    (y_test - x$forecast) / (y_test - pred_bench$forecast),
    2,
    function(r_t) {
      mean(abs(r_t))
    }
  )
}

#' Evaluate the Model Based on RelMAE (Relative MAE)
#' 
#' This function computes RelMAE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RelMAE vector corresponding to each variable.
#' @export
relmae <- function(x, pred_bench, y, ...) {
  UseMethod("relmae", x)
}

#' @rdname relmae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RelMAE implements MAE of benchmark model as relative measures.
#' Let \eqn{MAE_b} be the MAE of the benchmark model.
#' Then
#' 
#' \deqn{RelMAE = \frac{MAE}{MAE_b}}
#' 
#' where \eqn{MAE} is the MAE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' @export
relmae.predbvhar <- function(x, pred_bench, y, ...) {
  mae(x, y) / mae(pred_bench, y)
}

#' @rdname relmae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
relmae.bvharcv <- function(x, pred_bench, y, ...) {
  mae(x, y) / mae(pred_bench, y)
}

#' Evaluate the Model Based on RMAPE (Relative MAPE)
#' 
#' This function computes RMAPE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMAPE vector corresponding to each variable.
#' @export
rmape <- function(x, pred_bench, y, ...) {
  UseMethod("rmape", x)
}

#' @rdname rmape
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' RMAPE is the ratio of MAPE of given model and the benchmark one.
#' Let \eqn{MAPE_b} be the MAPE of the benchmark model.
#' Then
#' 
#' \deqn{RMAPE = \frac{mean(MAPE)}{mean(MAPE_b)}}
#' 
#' where \eqn{MAPE} is the MAPE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' @export
rmape.predbvhar <- function(x, pred_bench, y, ...) {
  mean(mape(x, y)) / mean(mape(pred_bench, y))
}

#' @rdname rmape
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmape.bvharcv <- function(x, pred_bench, y, ...) {
  mean(mape(x, y)) / mean(mape(pred_bench, y))
}

#' Evaluate the Model Based on RMASE (Relative MASE)
#' 
#' This function computes RMASE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMASE vector corresponding to each variable.
#' @export
rmase <- function(x, pred_bench, y, ...) {
  UseMethod("rmase", x)
}

#' @rdname rmase
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' RMASE is the ratio of MAPE of given model and the benchmark one.
#' Let \eqn{MASE_b} be the MAPE of the benchmark model.
#' Then
#' 
#' \deqn{RMASE = \frac{mean(MASE)}{mean(MASE_b)}}
#' 
#' where \eqn{MASE} is the MASE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' @export
rmase.predbvhar <- function(x, pred_bench, y, ...) {
  mean(mase(x, y)) / mean(mase(pred_bench, y))
}

#' @rdname rmase
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmase.bvharcv <- function(x, pred_bench, y, ...) {
  mean(mase(x, y)) / mean(mase(pred_bench, y))
}

#' Evaluate the Model Based on RMSFE
#' 
#' This function computes RMSFE (Mean Squared Forecast Error Relative to the Benchmark)
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMSFE vector corresponding to each variable.
#' @export
rmsfe <- function(x, pred_bench, y, ...) {
  UseMethod("rmsfe", x)
}

#' @rdname rmsfe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RMSFE is the ratio of L2 norm of \eqn{e_t} from forecasting object and from benchmark model.
#' 
#' \deqn{RMSFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)} \rVert)}}
#' 
#' where \eqn{e_t^{(b)}} is the error from the benchmark model.
#' @references 
#' Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
rmsfe.predbvhar <- function(x, pred_bench, y, ...) {
  sum(mse(x, y)) / sum(mse(pred_bench, y))
}

#' @rdname rmsfe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmsfe.bvharcv <- function(x, pred_bench, y, ...) {
  sum(mse(x, y)) / sum(mse(pred_bench, y))
}

#' Evaluate the Model Based on RMAFE
#' 
#' This function computes RMAFE (Mean Absolute Forecast Error Relative to the Benchmark)
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMAFE vector corresponding to each variable.
#' @export
rmafe <- function(x, pred_bench, y, ...) {
  UseMethod("rmafe", x)
}

#' @rdname rmafe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RMAFE is the ratio of L1 norm of \eqn{e_t} from forecasting object and from benchmark model.
#' 
#' \deqn{RMAFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)} \rVert)}}
#' 
#' where \eqn{e_t^{(b)}} is the error from the benchmark model.
#' 
#' @references 
#' Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
rmafe.predbvhar <- function(x, pred_bench, y, ...) {
  sum(mae(x, y)) / sum(mae(pred_bench, y))
}

#' @rdname rmafe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmafe.bvharcv <- function(x, pred_bench, y, ...) {
  sum(mae(x, y)) / sum(mae(pred_bench, y))
}

#' Evaluate the Model Based on Log Predictive Likelihood
#' 
#' This function computes LPL given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
lpl <- function(x, y, ...) {
  UseMethod("lpl", x)
}

#' @rdname lpl
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @references
#' Cross, J. L., Hou, C., & Poon, A. (2020). *Macroeconomic forecasting with large Bayesian VARs: Global-local priors and the illusion of sparsity*. International Journal of Forecasting, 36(3), 899–915.
#' 
#' Gruber, L., & Kastner, G. (2022). *Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!* arXiv.
#' @importFrom posterior as_draws_matrix
#' @export
lpl.predsv <- function(x, y, ...) {
  object <- x$object
  dim_data <- object$m
  h_record <- as_draws_matrix(object$h_record)
  compute_lpl(
    as.matrix(y),
    x$forecast,
    h_record[,(ncol(h_record) - dim_data + 1):ncol(h_record)],
    as_draws_matrix(object$a_record),
    as_draws_matrix(object$sigh_record)
  )
}
