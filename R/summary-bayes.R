#' Generate Coefficient Matrix and Covariance Matrix from Bayesian Model
#' 
#' @param x object
#' @param ... not used
#' 
#' @export
gen_posterior <- function(x, ...) {
  UseMethod("gen_posterior", x)
}

#' Posterior Distribution of Minnesota BVAR(p)
#' 
#' @description 
#' Generates Parameters of Minnesota BVAR \eqn{B, \Sigma_e}.
#' 
#' @param object \code{bvarmn} object
#' @param iter number to generate (By default, 100)
#' @param ... not used
#' 
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' \deqn{(B, \Sigma) \sim MNIW(\hat{B}, \hat{U}, \hat{\Sigma}, \alpha_0 + n + 2)}
#' 
#' @return \code{minnesota} object with:
#' \item{\code{coefficients}}{iter x k x m array: each column of the array indicate the draw for each lag corresponding to that variable}
#' \item{\code{covmat}}{iter x m x m array: each column of teh array indicate the draw for each varable corresponding to that variable}
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @importFrom mniw rmniw
#' @export
gen_posterior.bvarmn <- function(object, iter = 100, ...) {
  mn_mean <- object$mn_mean
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$a0 + object$obs + 2
  b_sig <- rmniw(n = iter, Lambda = mn_mean, Omega = mn_prec, Psi = iw_scale, nu = nu)
  Bhat <- b_sig$X
  Sighat <- b_sig$V
  # mniw returns list of 3d array---------
  dimnames(Bhat) <- list(
    rownames(mn_mean), # row
    colnames(mn_mean), # col
    1:iter # 3rd dim
  )
  dimnames(Sighat) <- list(
    rownames(iw_scale), # row
    colnames(iw_scale), # col
    1:iter # 3rd dim
  )
  res <- list(
    coefficients = Bhat,
    covmat = Sighat,
    N = iter
  )
  class(res) <- "minnesota"
  res
}

#' @inherit ggplot2::autolayer
#' @export
autolayer <- function(object, ...){
  UseMethod("autolayer")
}

#' autoplot
#' 
#' See \code{\link[ggplot2]{ggplot2::autoplot}}.
#' 
#' @importFrom ggplot2 autoplot
#' @name autoplot
#' @export
NULL

#' Density Plot for \code{minnesota} Object
#' 
#' @param object \code{minnesota} object
#' @param type Draw whether Mean or residuals
#' @param var_name variable name (for mean)
#' @param NROW Numer of facet row
#' @param NCOL Numer of facet col
#' @param ... not used
#' 
#' @importFrom ggplot2 ggplot aes geom_density geom_point facet_grid facet_wrap labs element_text element_blank
#' @importFrom tidyr pivot_longer
#' @importFrom tibble rownames_to_column
#' @export
autoplot.minnesota <- function(object, type = c("mean", "residual"), var_name = NULL, NROW = NULL, NCOL = NULL, ...) {
  type <- match.arg(type)
  X <- object$coefficients
  switch(type,
    "mean" = {
      if (is.null(var_name)) stop("Provide 'var_name'")
      X <- 
        lapply(
          1:(object$N),
          function(x) {
            X[,, x] %>% 
              as.data.frame() %>% 
              rownames_to_column(var = "lags")
          }
        ) %>% 
        bind_rows() %>% 
        pivot_longer(-lags, names_to = "name", values_to = "value")
      p <- 
        X %>% 
        ggplot(aes(x = value)) +
        geom_density() +
        facet_wrap(
          lags ~ .,
          nrow = NROW,
          ncol = NCOL,
          scales = "free"
        ) +
        labs(
          x = element_blank(),
          y = element_blank()
        )
      # p <- 
      #   data.frame(b = X[, lag_name, var_name]) %>% # 2d
      #   ggplot(aes(x = b)) +
      #   geom_density() +
      #   labs(
      #     title = element_text(title),
      #     x = element_blank(),
      #     y = element_blank()
      #   )
      p
    },
    "residual" = {
      # change this code later: residual Y0 - X0 Bhat
      X <- apply(X, 2, function(x) x) %>% as.data.frame() # 2d
      N <- nrow(X)
      X[["id"]] <- 1:N
      X <- 
        X %>% 
        pivot_longer(-id, names_to = "name", values_to = "value")
      p <- 
        X %>% 
        ggplot(aes(x = id, y = value)) +
        geom_point(alpha = .5) +
        facet_grid(name ~ ., scales = "free_y") +
        labs(
          x = element_blank(),
          y = element_blank()
        )
      p
    }
  )
}




