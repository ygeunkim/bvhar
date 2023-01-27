#' Evaluate the Estimation Based on Frobenius Norm
#' 
#' This function computes estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @export
fromse <- function(x, y, ...) {
  UseMethod("fromse", x)
}

#' @rdname fromse
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' Consider the Frobenius Norm \eqn{\lVert \cdot \rVert_F}.
#' let \eqn{\hat{\Phi}} be nrow x k the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{MSE = 100 \frac{\lVert \hat{\Phi} - \Phi \rVert_F}{nrow \times k}}
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
fromse.ssvsmod <- function(x, y, restrict = FALSE, ...) {
  if (restrict) {
    return(100 * norm(x$restricted_posterior - y, type = "F") / (x$df * x$m))
  }
  100 * norm(x$coefficients - y, type = "F") / (x$df * x$m)
}

#' @rdname fromse
#' @export
fromse.mvhsmod <- function(x, y, ...) {
  100 * norm(x$coefficients - y, type = "F") / (x$df * x$m)
}

#' Evaluate the Estimation Based on Spectral Norm Error
#' 
#' This function computes estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @export
spne <- function(x, y, ...) {
  UseMethod("spne", x)
}

#' @rdname spne
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{\lVert \hat{\Phi} - \Phi \rVert_2}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). doi:[10.1080/01621459.2018.1437043](https://doi.org/10.1080/01621459.2018.1437043)
#' @export
spne.ssvsmod <- function(x, y, restrict = FALSE, ...) {
  if (restrict) {
    return(norm(x$restricted_posterior - y, type = "2"))
  }
  norm(x$coefficients - y, type = "2")
}

#' @rdname spne
#' @export
spne.mvhsmod <- function(x, y, ...) {
  norm(x$coefficients - y, type = "2")
}

#' Evaluate the Estimation Based on Relative Spectral Norm Error
#' 
#' This function computes relative estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
relspne <- function(x, y, ...) {
  UseMethod("relspne", x)
}

#' @rdname relspne
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes relative estimation error by
#' \deqn{\frac{\lVert \hat{\Phi} - \Phi \rVert_2}{\lVert \Phi \rVert_2}}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). doi:[10.1080/01621459.2018.1437043](https://doi.org/10.1080/01621459.2018.1437043)
#' @export
relspne.ssvsmod <- function(x, y, restrict = FALSE, ...) {
  spne(x, y, restrict = restrict) / norm(y, type = "2")
}

#' @rdname relspne
#' @export
relspne.mvhsmod <- function(x, y, ...) {
  spne(x, y) / norm(y, type = "2")
}

#' Evaluate the Sparsity Estimation Based on Confusion Matrix
#' 
#' This function computes FDR (false discovery rate) and FNR (false negative rate) for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param ... not used
#' @export
confusion <- function(x, y, threshold, ...) {
  UseMethod("confusion", x)
}

#' @rdname confusion
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' If the element of the estimate \eqn{\hat\Phi} is smaller than some threshold,
#' it is treated to be zero.
#' 
#' In this confusion matrix, positive (0) means sparsity.
#' FP is false positive, and TP is true positive.
#' FN is false negative, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
confusion.ssvsmod <- function(x, y, threshold = .01, truth_thr = 0, restrict = FALSE, ...) {
  if (restrict) {
    est <- c(x$restricted_posterior)
  } else {
    est <- c(x$coefficients)
  }
  est <- ifelse(abs(est) <= threshold, 0, 1)
  table(truth = ifelse(c(y) <= truth_thr, 0, 1), estimation = est)
}

#' @rdname confusion
#' @export
confusion.mvhsmod <- function(x, y, threshold = .01, truth_thr = 0, ...) {
  est <- c(x$coefficients)
  est <- ifelse(abs(est) <= threshold, 0, est)
  est <- ifelse(abs(est) <= threshold, 0, 1)
  table(truth = ifelse(c(y) <= truth_thr, 0, 1), estimation = est)
}

#' Evaluate the Sparsity Estimation Based on Precision
#' 
#' This function computes precision for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @export
conf_prec <- function(x, y, threshold, ...) {
  UseMethod("conf_prec", x)
}

#' @rdname conf_prec
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' If the element of the estimate \eqn{\hat\Phi} is smaller than some threshold,
#' it is treated to be zero.
#' Then the precision is computed by
#' \deqn{precision = \frac{TP}{TP + FP}}
#' where TP is true positive, and FP is false positive.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
conf_prec.ssvsmod <- function(x, y, threshold = .01, truth_thr = 0, restrict = FALSE, ...) {
  conftab <- confusion(x, y, threshold = threshold, truth_thr = truth_thr, restrict = restrict)
  conftab[1, 1] / (conftab[1, 1] + conftab[2, 1])
}

#' @rdname conf_prec
#' @export
conf_prec.mvhsmod <- function(x, y, threshold = .01, truth_thr = 0, ...) {
  conftab <- confusion(x, y, threshold = threshold, truth_thr = truth_thr)
  conftab[1, 1] / (conftab[1, 1] + conftab[2, 1])
}

#' Evaluate the Sparsity Estimation Based on Recall
#' 
#' This function computes recall for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param ... not used
#' @export
conf_recall <- function(x, y, threshold, ...) {
  UseMethod("conf_recall", x)
}

#' @rdname conf_recall
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' If the element of the estimate \eqn{\hat\Phi} is smaller than some threshold,
#' it is treated to be zero.
#' Then the precision is computed by
#' \deqn{recall = \frac{TP}{TP + FN}}
#' where TP is true positive, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
conf_recall.ssvsmod <- function(x, y, threshold = .01, truth_thr = 0, restrict = FALSE, ...) {
  conftab <- confusion(x, y, threshold = threshold, truth_thr = truth_thr, restrict = restrict)
  conftab[1, 1] / (conftab[1, 1] + conftab[1, 2])
}

#' @rdname conf_recall
#' @export
conf_recall.mvhsmod <- function(x, y, threshold = .01, truth_thr = 0, ...) {
  conftab <- confusion(x, y, threshold = threshold, truth_thr = truth_thr)
  conftab[1, 1] / (conftab[1, 1] + conftab[1, 2])
}

#' Evaluate the Sparsity Estimation Based on F1 Score
#' 
#' This function computes F1 score for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param ... not used
#' @export
conf_fscore <- function(x, y, threshold, ...) {
  UseMethod("conf_fscore", x)
}

#' @rdname conf_fscore
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param threshold Threshold value indicating sparsity.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param restrict Use restricted VAR. By default, `FALSE`.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' If the element of the estimate \eqn{\hat\Phi} is smaller than some threshold,
#' it is treated to be zero.
#' Then the F1 score is computed by
#' \deqn{F_1 = \frac{2 precision \times recall}{precision + recall}}
#' @export
conf_fscore.ssvsmod <- function(x, y, threshold = .01, truth_thr = 0, restrict = FALSE, ...) {
  prec_score <- conf_prec(x, y, threshold = threshold, truth_thr = truth_thr, restrict = restrict)
  rec_score <- conf_recall(x, y, threshold = threshold, truth_thr = truth_thr, restrict = restrict)
  2 * prec_score * rec_score / (prec_score + rec_score)
}

#' @rdname conf_fscore
#' @export
conf_fscore.mvhsmod <- function(x, y, threshold = .01, truth_thr = 0, ...) {
  prec_score <- conf_prec(x, y, threshold = threshold, truth_thr = truth_thr)
  rec_score <- conf_recall(x, y, threshold = threshold, truth_thr = truth_thr)
  2 * prec_score * rec_score / (prec_score + rec_score)
}
