#' Summarizing VAR and VHAR with SSVS Prior Model
#' 
#' Conduct variable selection.
#' 
#' @param object `ssvsmod` object
#' @param coef_threshold Threshold for variable selection. By default, `0.5`.
#' @param chol_threshold Threshold for variable selection in cholesky factor. By default, `0.5`.
#' @param ... not used
#' @details 
#' In each cell, variable selection can be done by giving threshold for posterior mean of \eqn{\gamma}.
#' 
#' @importFrom stats coef
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889. doi:[10.1080/01621459.1993.10476353](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476353)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' 
#' O’Hara, R. B., & Sillanpää, M. J. (2009). *A review of Bayesian variable selection methods: what, how and which*. Bayesian Analysis, 4(1), 85–117. doi:[10.1214/09-ba403](https://doi.org/10.1214/09-BA403)
#' @export
summary.ssvsmod <- function(object, coef_threshold = .5, chol_threshold = .5, ...) {
  # coefficients-------------------------------
  coef_mean <- object$coefficients
  coef_dummy <- object$pip
  var_selection <- coef_dummy > coef_threshold
  coef_res <- ifelse(var_selection, coef_mean, 0L)
  rownames(coef_res) <- rownames(coef_mean)
  colnames(coef_res) <- colnames(coef_mean)
  # cholesky factor----------------------------
  chol_mean <- object$chol_posterior
  chol_dummy <- object$omega_posterior
  chol_selection <- chol_dummy > chol_threshold
  chol_res <- ifelse(chol_selection, chol_mean, 0L)
  # return S3 object---------------------------
  res <- list(
    call = object$call,
    process = object$process,
    p = object$p,
    m = object$m,
    type = object$type,
    coefficients = coef_res,
    cholesky = chol_res,
    threshold = c(coef_threshold, chol_threshold),
    coef_choose = var_selection,
    chol_choose = chol_selection
  )
  class(res) <- c("summary.ssvsmod", "summary.bvharsp")
  res
}

#' Summarizing VAR and VHAR with Matrix-variate Horseshoe Prior Model
#' 
#' Conduct variable selection.
#' 
#' @param object `mvhsmod` object
#' @param level Specify alpha of credible interval level 100(1 - alpha) percentage. By default, `.05`.
#' @param ... not used
#' @details 
#' MCMC can construct \eqn{100 (1 - \alpha)} credible interval.
#' This interval can help variable selection.
#' 
#' @importFrom posterior summarise_draws subset_draws
#' @importFrom stats quantile
#' @importFrom dplyr rename mutate
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
summary.mvhsmod <- function(object, level = .05, ...) {
  cred_int <- 
    object$param %>% 
    subset_draws("alpha|phi", regex = TRUE) %>% 
    summarise_draws(
      ~quantile(
        .,
        prob = c(level / 2, (1 - level / 2))
      )
    ) %>% 
    rename("term" = "variable", "conf.low" = `2.5%`, "conf.high" = `97.5%`)
  selection <- matrix(ifelse(cred_int$conf.low * cred_int$conf.high < 0, FALSE, TRUE), ncol = object$m)
  coef_res <- ifelse(selection, object$coefficients, 0L)
  rownames(selection) <- rownames(object$coefficients)
  colnames(selection) <- colnames(object$coefficients)
  rownames(coef_res) <- rownames(object$coefficients)
  colnames(coef_res) <- colnames(object$coefficients)
  # return S3 object---------------------------
  res <- list(
    call = object$call,
    process = object$process,
    p = object$p,
    m = object$m,
    type = object$type,
    interval = cred_int,
    coefficients = coef_res,
    level = level,
    coef_choose = selection
  )
  class(res) <- c("summary.mvhsmod", "summary.bvharsp")
  res
}

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
#' @param ... not used
#' @details 
#' Consider the Frobenius Norm \eqn{\lVert \cdot \rVert_F}.
#' let \eqn{\hat{\Phi}} be nrow x k the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{MSE = 100 \frac{\lVert \hat{\Phi} - \Phi \rVert_F}{nrow \times k}}
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
fromse.bvharsp <- function(x, y, ...) {
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
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{\lVert \hat{\Phi} - \Phi \rVert_2}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). doi:[10.1080/01621459.2018.1437043](https://doi.org/10.1080/01621459.2018.1437043)
#' @export
spne.bvharsp <- function(x, y, ...) {
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
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes relative estimation error by
#' \deqn{\frac{\lVert \hat{\Phi} - \Phi \rVert_2}{\lVert \Phi \rVert_2}}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). doi:[10.1080/01621459.2018.1437043](https://doi.org/10.1080/01621459.2018.1437043)
#' @export
relspne.bvharsp <- function(x, y, ...) {
  spne(x, y) / norm(y, type = "2")
}

#' Evaluate the Sparsity Estimation Based on Confusion Matrix
#' 
#' This function computes FDR (false discovery rate) and FNR (false negative rate) for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
confusion <- function(x, y, ...) {
  UseMethod("confusion", x)
}

#' @rdname confusion
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' 
#' In this confusion matrix, positive (0) means sparsity.
#' FP is false positive, and TP is true positive.
#' FN is false negative, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
confusion.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  est <- ifelse(c(x$coef_choose), 1L, 0L) %>% factor(levels = c(0L, 1L))
  truth <- ifelse(c(y) <= truth_thr, 0L, 1L) %>% factor(levels = c(0L, 1L))
  table(truth = truth, estimation = est)
}

#' Evaluate the Sparsity Estimation Based on FDR
#' 
#' This function computes false discovery rate (FDR) for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
conf_fdr <- function(x, y, ...) {
  UseMethod("conf_fdr", x)
}

#' @rdname conf_fdr
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' False discovery rate (FDR) is computed by
#' \deqn{FDR = \frac{FP}{TP + FP}}
#' where TP is true positive, and FP is false positive.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
conf_fdr.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[2, 1] / (conftab[1, 1] + conftab[2, 1])
}

#' Evaluate the Sparsity Estimation Based on Precision
#' 
#' This function computes precision for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
conf_prec <- function(x, y, ...) {
  UseMethod("conf_prec", x)
}

#' @rdname conf_prec
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
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
conf_prec.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 1] / (conftab[1, 1] + conftab[2, 1])
}

#' Evaluate the Sparsity Estimation Based on FNR
#' 
#' This function computes false negative rate (FNR) for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
conf_fnr <- function(x, y, ...) {
  UseMethod("conf_fnr", x)
}

#' @rdname conf_fnr
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' False negative rate (FNR) is computed by
#' \deqn{FNR = \frac{FN}{TP + FN}}
#' where TP is true positive, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
conf_fnr.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 2] / (conftab[1, 1] + conftab[1, 2])
}

#' Evaluate the Sparsity Estimation Based on Recall
#' 
#' This function computes recall for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
conf_recall <- function(x, y, ...) {
  UseMethod("conf_recall", x)
}

#' @rdname conf_recall
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' Precision is computed by
#' \deqn{recall = \frac{TP}{TP + FN}}
#' where TP is true positive, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' @export
conf_recall.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 1] / (conftab[1, 1] + conftab[1, 2])
}

#' Evaluate the Sparsity Estimation Based on F1 Score
#' 
#' This function computes F1 score for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param ... not used
#' @export
conf_fscore <- function(x, y, ...) {
  UseMethod("conf_fscore", x)
}

#' @rdname conf_fscore
#' @param x `summary.bvharsp` object.
#' @param y True coefficient matrix.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' Then the F1 score is computed by
#' \deqn{F_1 = \frac{2 precision \times recall}{precision + recall}}
#' @export
conf_fscore.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  prec_score <- conf_prec(x, y, truth_thr = truth_thr)
  rec_score <- conf_recall(x, y, truth_thr = truth_thr)
  2 * prec_score * rec_score / (prec_score + rec_score)
}
