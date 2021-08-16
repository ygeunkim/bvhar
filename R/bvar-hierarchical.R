#' Fit Bayesian VAR(p) of Hierarchical Normal-Mixture Prior
#' 
#' @description
#' This function fits BVAR(p) with Ghosh et al. (2018) hierarchical priors.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param iter The number of iterations for Gibbs sampler
#' @param type Prior types ("wishart": default, "lasso": Bayesian Group Lasso, "t": Multivariate t-distribution). See the detail.
#' @param Psi Scale matrix for Wishart prior when \code{type = "wishart"}
#' @param d Degrees of freedom of Wishart prior becomes d + k when \code{type = "wishart"}. d > -(k + 1)
#' 
#' @details 
#' Ghosh et al. (2018) gives flat prior for residual matrix in BVAR.
#' 
#' Under this setting, there are many models such as hierarchical or non-hierarchical.
#' This function chooses hierarchical prior in Section 3.3.
#' 
#' Non-hierarchical prior should pre-specify the precision matrix of matrix normal,
#' while it has prior distribution in hierarchical model.
#' 
#' As in the paper, this function provides the three types of hyper-prior,
#' 
#' \enumerate{
#'   \item Wishart prior: \eqn{U \sim W(\Psi, d + k)}, Psi and d prespecified
#'   \item Bayesian Group Lasso
#'   \item Multivariate t-Distribution
#' }
#' 
#' @return \code{bvar_mixture} returns an object \code{bvarghosh} \link{class}.
#' 
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: Ghosh}
#'   \item{call}{Matched call}
#'   \item{mn_mean}{Location of posterior matrix normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Precision matrix of posterior matrix normal distribution}
#'   \item{iw_scale}{Scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Shape of posterior inverse-wishart distribution}
#' }
#' 
#' @references 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526). \url{https://doi:10.1080/01621459.2018.1437043}
#' 
#' @importFrom mniw rwish
#' @order 1
#' @export
bvar_mixture <- function(y, p, iter = 1000L, type = c("wishart", "lasso", "t"), Psi = NULL, d = NULL) {
  if (!is.matrix(y)) y <- as.matrix(y)
  type <- match.arg(type)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  # initialize------------------------
  switch(
    type,
    "wishart" = {
      k <- ncol(X0)
      if (missing(Psi)) Psi <- diag(k)
      if (missing(d)) d <- 0
      # initialize U
      U <- rwish(Psi, d + k)
      # initialize B and Sigma
      init <- estimate_mn_flat(X0, Y0, U)
      B <- init$bhat
      Sig <- init$iwscale
      # Block gibbs
      
      # fill later
      name_var
    },
    "lasso" = {
      # fill later
      name_var
    },
    "t" = {
      # fill later
      name_var
    }
  )
}