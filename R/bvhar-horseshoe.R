#' Fitting Bayesian VHAR of Horseshoe Prior
#' 
#' `r lifecycle::badge("experimental")` This function fits VHAR with horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec Horseshoe initialization specification by [set_horseshoe()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Minnesota type
#' @param algo Ordinary gibbs sampling (`"gibbs"`) or blocked gibbs (Default: `"block"`).
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @return `bvhar_horseshoe` returns an object named `bvarhs` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior mean of VHAR coefficients.}
#'   \item{covmat}{Posterior mean of covariance matrix}
#'   \item{psi_posterior}{Posterior mean of precision matrix \eqn{\Psi}}
#'   \item{param}{[posterior::draws_df] with every variable: alpha, lambda, tau, omega, and eta}
#'   \item{param_names}{Name of every parameter.}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VHAR_Horseshoe"`}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{algo}{Usual Gibbs sampling (`"gibbs"`) or fast sampling (`"fast"`)}
#'   \item{spec}{Horseshoe specification defined by [set_horseshoe()]}
#'   \item{chain}{The numer of chains}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{group}{Indicators for group.}
#'   \item{num_group}{Number of groups.}
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
                            num_chains = 1,
                            num_iter = 1000, 
                            num_burn = floor(num_iter / 2),
                            thinning = 1,
                            bayes_spec = set_horseshoe(),
                            include_mean = TRUE,
                            minnesota = c("no", "short", "longrun"),
                            algo = c("block", "gibbs"),
                            verbose = FALSE,
                            num_thread = 1) {
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
  Y0 <- build_response(y, har[2], har[2] + 1)
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
  if (length(bayes_spec$local_sparsity) != num_restrict) {
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
  glob_idmat <- build_grpmat(
    p = 3,
    dim_data = dim_data,
    dim_design = dim_har,
    num_coef = num_restrict,
    minnesota = minnesota,
    include_mean = include_mean
  )
  grp_id <- unique(c(glob_idmat))
  global_sparsity <- rep(bayes_spec$global_sparsity, length(grp_id))
  # MCMC-----------------------------
  num_design <- nrow(Y0)
  fast <- FALSE
  if (num_design <= num_restrict) {
    fast <- TRUE
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  res <- estimate_sur_horseshoe(
    num_chains = num_chains,
    num_iter = num_iter,
    num_burn = num_burn,
    thin = thinning,
    x = X1,
    y = Y0,
    init_local = bayes_spec$local_sparsity,
    init_global = global_sparsity,
    init_sigma = 1,
    grp_id = grp_id,
    grp_mat = glob_idmat,
    blocked_gibbs = algo,
    fast = fast,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    display_progress = verbose,
    nthreads = num_thread
  )
  res <- do.call(rbind, res)
  colnames(res) <- gsub(pattern = "^alpha", replacement = "phi", x = colnames(res)) # alpha to phi
  rec_names <- colnames(res) # *_record
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
  res <- apply(
    res,
    2,
    function(x) {
      if (is.vector(x[[1]])) {
        return(as.matrix(unlist(x)))
      }
      do.call(rbind, x)
    }
  )
  names(res) <- rec_names # *_record
  res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  res$covmat <- mean(res$sigma) * diag(dim_data)
  res$psi_posterior <- diag(dim_data) / mean(res$sigma)
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  colnames(res$psi_posterior) <- name_var
  rownames(res$psi_posterior) <- name_var
  res$pip <- matrix(colMeans(res$kappa_record), ncol = dim_data)
  colnames(res$pip) <- name_var
  rownames(res$pip) <- name_har
  # preprocess the results-----------
  if (num_chains > 1) {
    res[rec_names] <- lapply(
      seq_along(res[rec_names]),
      function(id) {
        split_chain(res[rec_names][[id]], chain = num_chains, varname = param_names[id])
      }
    )
  } else {
    res[rec_names] <- lapply(
      seq_along(res[rec_names]),
      function(id) {
        colnames(res[rec_names][[id]]) <- paste0(param_names[id], "[", seq_len(ncol(res[rec_names][[id]])), "]")
        res[rec_names][[id]]
      }
    )
  }
  res[rec_names] <- lapply(res[rec_names], as_draws_df)
  # Parameters-----------------
  res$param <- bind_draws(
    res$phi_record,
    res$lambda_record,
    res$tau_record,
    res$sigma_record,
    res$kappa_record
  )
  res[rec_names] <- NULL
  res$param_names <- param_names
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
  res$chain <- num_chains
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
