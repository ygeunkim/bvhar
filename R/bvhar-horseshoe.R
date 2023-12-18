#' Fitting Bayesian VHAR of Horseshoe Prior
#' 
#' This function fits VHAR with horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec Horseshoe initialization specification by [set_horseshoe()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Minnesota type
#' @param algo Ordinary gibbs sampling (`"gibbs"`) or blocked gibbs (Default: `"block"`).
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @return `bvhar_horseshoe` returns an object named `bvarhs` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{phi_record}{MCMC trace for vectorized coefficients (alpha \eqn{\phi}) with [posterior::draws_df] format.}
#'   \item{lambda_record}{MCMC trace for local shrinkage level (lambda \eqn{\lambda}) with [posterior::draws_df] format.}
#'   \item{tau_record}{MCMC trace for global shrinkage level (tau \eqn{\tau}) with [posterior::draws_df] format.}
#'   \item{psi_record}{MCMC trace for precision matrix (psi \eqn{\Psi}) with [list] format.}
#'   \item{chain}{The numer of chains}
#'   \item{coefficients}{Posterior mean of VHAR coefficients.}
#'   \item{psi_posterior}{Posterior mean of precision matrix \eqn{\Psi}}
#'   \item{covmat}{Posterior mean of covariance matrix}
#'   \item{omega_record}{MCMC trace for diagonal element of \eqn{\Psi} (omega) with [posterior::draws_df] format.}
#'   \item{eta_record}{MCMC trace for upper triangular element of \eqn{\Psi} (eta) with [posterior::draws_df] format.}
#'   \item{param}{[posterior::draws_df] with every variable: alpha, lambda, tau, omega, and eta}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VHAR_Horseshoe"`}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{algo}{Usual Gibbs sampling (`"gibbs"`) or fast sampling (`"fast"`)}
#'   \item{spec}{Horseshoe specification defined by [set_horseshoe()]}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' Kim, Y. G., and Baek, C. (2023). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#'
#' Kim, Y. G., and Baek, C. (n.d.). Working paper.
#' @importFrom posterior as_draws_df bind_draws
#' @importFrom stats cov
#' @order 1
#' @export
bvhar_horseshoe <- function(y,
                            har = c(5, 22),
                            num_iter = 1000, 
                            num_burn = floor(num_iter / 2),
                            thinning = 1,
                            bayes_spec = set_horseshoe(),
                            include_mean = TRUE,
                            minnesota = c("no", "short", "longrun"),
                            algo = c("block", "gibbs"),
                            verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  minnesota <- match.arg(minnesota)
  algo <- match.arg(algo)
  algo <- switch(algo, "gibbs" = 1, "block" = 2)
  # model specification---------------
  if (!is.horseshoespec(bayes_spec)) {
    stop("Provide 'horseshoespec' for 'bayes_spec'.")
  }
  bayes_spec$process <- "VHAR"
  # MCMC iterations-------------------
  if (num_iter < 1) {
    stop("Iterate more than 1 times for MCMC.")
  }
  if (num_iter < num_burn) {
    stop("'num_iter' should be larger than 'num_burn'.")
  }
  if (thinning < 1) {
    stop("'thinning' should be non-negative.")
  }
  # Y0 = X1 Phi + Z-------------------
  Y0 <- build_y0(y, har[2], har[2] + 1)
  dim_data <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, har[2], include_mean)
  colnames(X0) <- concatenate_colnames(name_var, 1:har[2], include_mean)
  hartrans_mat <- scale_har(dim_data, har[1], har[2], include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean)
  X1 <- X0 %*% t(hartrans_mat)
  colnames(X1) <- name_har
  # Initial vectors-------------------
  dim_har <- ncol(X1)
  num_restrict <- ifelse(
    include_mean,
    dim_data^2 * 3 + dim_data,
    dim_data^2 * 3
  )
  if (length(bayes_spec$local_sparsity) != dim_har) {
    if (length(bayes_spec$local_sparsity) == 1) {
      bayes_spec$local_sparsity <- rep(bayes_spec$local_sparsity, num_restrict)
    } else {
      stop("Length of the vector 'local_sparsity' should be dim * 3 or dim * 3 + 1.")
    }
  }
  if (include_mean) {
    idx <- c(gl(3, dim_data), 4)
  } else {
    idx <- gl(3, dim_data)
  }
  glob_idmat <- switch(
    minnesota,
    "no" = matrix(1L, nrow = dim_har, ncol = dim_data),
    "short" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_restrict), ncol = dim_data),
        idx
      )
      glob_idmat[[1]] <- diag(dim_data) + 1
      id <- 1
      for (i in 2:3) {
        glob_idmat[[i]] <- matrix(i + 1, nrow = dim_data, ncol = dim_data)
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    },
    "longrun" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_restrict), ncol = dim_data),
        idx
      )
      id <- 1
      for (i in 1:3) {
        glob_idmat[[i]] <- diag(dim_data) + id
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    }
  )
  grp_id <- unique(c(glob_idmat[1:(dim_data * 3),]))
  global_sparsity <- rep(bayes_spec$global_sparsity, length(grp_id))
  # MCMC-----------------------------
  num_design <- nrow(Y0)
  fast <- FALSE
  if (num_design <= num_restrict) {
    fast <- TRUE
  }
  res <- estimate_sur_horseshoe(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X1,
    y = Y0,
    init_local = bayes_spec$local_sparsity,
    init_global = global_sparsity,
    init_sigma = 1,
    grp_id = grp_id,
    grp_mat = glob_idmat,
    blocked_gibbs = algo,
    fast = fast,
    display_progress = verbose
  )
  # preprocess the results-----------
  names(res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(res))
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$phi_record <- res$phi_record[thin_id,]
  colnames(res$phi_record) <- paste0("phi[", seq_len(ncol(res$phi_record)), "]")
  res$coefficients <- 
    colMeans(res$phi_record) %>% 
    matrix(ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  res$phi_record <- as_draws_df(res$phi_record)
  if (minnesota == "no") {
    res$tau_record <- as.matrix(res$tau_record[thin_id])
    colnames(res$tau_record) <- "tau"
  } else {
    res$tau_record <- res$tau_record[thin_id,]
    colnames(res$tau_record) <- paste0(
      "tau[",
      seq_len(ncol(res$tau_record)),
      "]"
    )
  }
  res$tau_record <- as_draws_df(res$tau_record)
  res$lambda_record <- res$lambda_record[thin_id,]
  colnames(res$lambda_record) <- paste0(
    "lambda[",
    seq_len(ncol(res$lambda_record)),
    "]"
  )
  res$lambda_record <- as_draws_df(res$lambda_record)
  res$covmat <- mean(res$sigma) * diag(dim_data)
  res$psi_posterior <- diag(dim_data) / mean(res$sigma)
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  colnames(res$psi_posterior) <- name_var
  rownames(res$psi_posterior) <- name_var
  res$sigma_record <- as.matrix(res$sigma_record[thin_id])
  colnames(res$sigma_record) <- "sigma"
  res$sigma_record <- as_draws_df(res$sigma_record)
  res$kappa_record <- res$kappa_record[thin_id,]
  colnames(res$kappa_record) <- paste0(
    "kappa[",
    seq_len(ncol(res$kappa_record)),
    "]"
  )
  res$pip <- matrix(colMeans(res$kappa_record), ncol = dim_data)
  colnames(res$pip) <- name_var
  rownames(res$pip) <- name_har
  res$kappa_record <- as_draws_df(res$kappa_record)
  # Parameters-----------------
  res$param <- bind_draws(
    res$phi_record,
    res$lambda_record,
    res$tau_record,
    res$sigma_record
  )
  # variables------------
  res$df <- ncol(X0)
  res$p <- 3
  res$week <- har[1]
  res$month <- har[2]
  res$m <- ncol(y)
  res$obs <- num_design
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VHAR", bayes_spec$prior, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$algo <- ifelse(algo == 1, "gibbs", "blocked")
  res$spec <- bayes_spec
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  res$group <- glob_idmat
  res$num_group <- length(grp_id)
  # data------------------
  res$HARtrans <- hartrans_mat
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  # return S3 object-----------------
  class(res) <- c("bvharhs", "hsmod", "bvharsp")
  res
}
