#' Fitting Bayesian VHAR of SSVS Prior
#' 
#' This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of warm-up (burn-in). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A SSVS model specification by [set_ssvs()]. By default, use a default semiautomatic approach [choose_ssvs()].
#' @param init_spec SSVS initialization specification by [init_ssvs()]. By default, use OLS for coefficient and cholesky factor while 1 for dummies.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). Two type: `"short"` type and `"longrun"` type. By default, `"no"`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @details 
#' SSVS prior gives prior to parameters \eqn{\alpha = vec(A)} (VAR coefficient) and \eqn{\Sigma_e^{-1} = \Psi \Psi^T} (residual covariance).
#' 
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \kappa_{0j}^2) + \gamma_j N(0, \kappa_{1j}^2)}
#' \deqn{\gamma_j \sim Bernoulli(q_j)}
#' 
#' and for upper triangular matrix \eqn{\Psi},
#' 
#' \deqn{\psi_{jj}^2 \sim Gamma(shape = a_j, rate = b_j)}
#' \deqn{\psi_{ij} \mid w_{ij} \sim (1 - w_{ij}) N(0, \kappa_{0,ij}^2) + w_{ij} N(0, \kappa_{1,ij}^2)}
#' \deqn{w_{ij} \sim Bernoulli(q_{ij})}
#' 
#' Gibbs sampler is used for the estimation.
#' See [ssvs_bvar_algo] how it works.
#' @return `bvhar_ssvs` returns an object named `bvharssvs` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{phi_record}{MCMC trace for vectorized coefficients (phi \eqn{\phi}) with [posterior::draws_df] format.}
#'   \item{eta_record}{MCMC trace for upper triangular element of cholesky factor (eta \eqn{\eta}) with [posterior::draws_df] format.}
#'   \item{psi_record}{MCMC trace for diagonal element of cholesky factor (psi \eqn{\psi}) with [posterior::draws_df] format.}
#'   \item{omega_record}{MCMC trace for indicator variable for \eqn{eta} (omega \eqn{\omega}) with [posterior::draws_df] format.}
#'   \item{gamma_record}{MCMC trace for indicator variable for \eqn{alpha} (gamma \eqn{\gamma}) with [posterior::draws_df] format.}
#'   \item{chol_record}{MCMC trace for cholesky factor matrix \eqn{\Psi} with [list] format.}
#'   \item{ols_coef}{OLS estimates for VAR coefficients.}
#'   \item{ols_cholesky}{OLS estimates for cholesky factor}
#'   \item{coefficients}{Posterior mean of VAR coefficients.}
#'   \item{omega_posterior}{Posterior mean of omega}
#'   \item{pip}{Posterior inclusion probability}
#'   \item{param}{[posterior::draws_df] with every variable: alpha, eta, psi, omega, and gamma}
#'   \item{chol_posterior}{Posterior mean of cholesky factor matrix}
#'   \item{covmat}{Posterior mean of covariance matrix}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VHAR_SSVS"`}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{spec}{SSVS specification defined by [set_ssvs()]}
#'   \item{init}{Initial specification defined by [init_ssvs()]}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{chain}{The numer of chains}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' Kim, Y. G., and Baek, C. (2023). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#'
#' Kim, Y. G., and Baek, C. (n.d.). Working paper.
#' @seealso 
#' * Vectorization formulation [var_vec_formulation]
#' * Gibbs sampler algorithm [ssvs_bvar_algo]
#' @importFrom posterior as_draws_df bind_draws
#' @importFrom foreach foreach getDoParRegistered
#' @importFrom doRNG %dorng%
#' @order 1
#' @export
bvhar_ssvs <- function(y, 
                       har = c(5, 22), 
                       num_iter = 1000, 
                       num_burn = floor(num_iter / 2), 
                       thinning = 1,
                       bayes_spec = choose_ssvs(y = y, ord = har, type = "VHAR", param = c(.1, 10), include_mean = include_mean, gamma_param = c(.01, .01), mean_non = 0, sd_non = .1),
                       init_spec = init_ssvs(type = "auto"),
                       include_mean = TRUE,
                       minnesota = c("no", "short", "longrun"),
                       verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  minnesota <- match.arg(minnesota)
  # model specification---------------
  if (!is.ssvsinput(bayes_spec)) {
    stop("Provide 'ssvsinput' for 'bayes_spec'.")
  }
  if (!is.ssvsinit(init_spec)) {
    stop("Provide 'ssvsinit' for 'init_spec'.")
  }
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
  dim_data <- ncol(y) # k
  # dim_har <- 3 * dim_data + 1
  Y0 <- build_y0(y, har[2], har[2] + 1) # n x k
  num_design <- nrow(Y0) # n
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, har[2], include_mean) # n x dim_design
  colnames(X0) <- concatenate_colnames(name_var, 1:har[2], include_mean)
  hartrans_mat <- scale_har(dim_data, har[1], har[2], include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean)
  X1 <- X0 %*% t(hartrans_mat)
  colnames(X1) <- name_har
  dim_har <- ncol(X1)
  # no regularization for diagonal term---------------------
  num_restrict <- 3 * dim_data^2 # restrict only coefficients
  glob_idmat <- switch(
    minnesota,
    "no" = matrix(1L, nrow = num_restrict / dim_data, ncol = dim_data),
    "short" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_restrict), ncol = dim_data),
        gl(3, dim_data)
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
        gl(3, dim_data)
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
  num_grp <- length(grp_id)
  # length 1 of bayes_spec--------------
  num_eta <- dim_data * (dim_data - 1) / 2 # number of upper element of Psi
  if (length(bayes_spec$coef_spike) == 1) {
    bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_restrict)
  }
  if (length(bayes_spec$coef_slab) == 1) {
    bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_restrict)
  }
  if (length(bayes_spec$coef_mixture) == 1) {
    bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_grp)
  }
  # if (length(bayes_spec$mean_coef) == 1) {
  #   bayes_spec$mean_coef <- rep(bayes_spec$mean_coef, num_restrict)
  # }
  if (length(bayes_spec$mean_non) == 1) {
    bayes_spec$mean_non <- rep(bayes_spec$mean_non, dim_data)
  }
  
  if (length(bayes_spec$shape) == 1) {
    bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
  }
  if (length(bayes_spec$rate) == 1) {
    bayes_spec$rate <- rep(bayes_spec$rate, dim_data)
  }
  if (length(bayes_spec$chol_spike) == 1) {
    bayes_spec$chol_spike <- rep(bayes_spec$chol_spike, num_eta)
  }
  if (length(bayes_spec$chol_slab) == 1) {
    bayes_spec$chol_slab <- rep(bayes_spec$chol_slab, num_eta)
  }
  if (length(bayes_spec$chol_mixture) == 1) {
    bayes_spec$chol_mixture <- rep(bayes_spec$chol_mixture, num_eta)
  }
  if (all(is.na(bayes_spec$coef_spike)) || all(is.na(bayes_spec$coef_slab))) {
    # Conduct semiautomatic function using var_lm()
    stop("Specify spike-and-slab of coefficients.")
  }
  if (all(is.na(bayes_spec$chol_spike)) || all(is.na(bayes_spec$chol_slab))) {
    # Conduct semiautomatic function using var_lm()
    stop("Specify spike-and-slab of cholesky factor.")
  }
  # Error----------------------------
  if (!(
    length(bayes_spec$coef_spike) == num_restrict &&
    length(bayes_spec$coef_slab) == num_restrict &&
    length(bayes_spec$coef_mixture) == num_grp
    # && length(bayes_spec$mean_coef) == num_restrict
  )) {
    stop("Invalid 'coef_spike', 'coef_slab', and 'coef_mixture' size. The vector size should be the same as 3 * dim^2.")
  }
  if (!(length(bayes_spec$shape) == dim_data && length(bayes_spec$rate) == dim_data)) {
    stop("Size of SSVS 'shape' and 'rate' vector should be the same as the time series dimension.")
  }
  if (!(
    length(bayes_spec$chol_spike) == num_eta &&
    length(bayes_spec$chol_slab) == length(bayes_spec$chol_spike) &&
    length(bayes_spec$chol_mixture) == length(bayes_spec$chol_spike)
  )) {
    stop("Invalid 'chol_spike', 'chol_slab', and 'chol_mixture' size. The vector size should be the same as dim * (dim - 1) / 2.")
  }
  # Initial vectors-------------------
  if (init_spec$chain == 1) {
    if (init_spec$type == "user") {
      if (!(nrow(init_spec$init_coef) == dim_har || ncol(init_spec$init_coef) == dim_data)) {
        stop("Dimension of 'init_coef' should be (3 * dim) x dim or (3 * dim + 1) x dim.")
      }
      if (!(nrow(init_spec$init_coef_dummy) == num_restrict || ncol(init_spec$init_coef_dummy) == dim_data)) {
        stop("Dimension of 'init_coef_dummy' should be (dim * p) x dim x dim.")
      }
      init_coef <- c(init_spec$init_coef)
      init_coef_dummy <- c(init_spec$init_coef_dummy)
      init_chol_diag <- diag(init_spec$init_chol)
      init_chol_upper <- init_spec$init_chol[upper.tri(init_spec$init_chol, diag = FALSE)]
      init_chol_dummy <- init_spec$init_chol_dummy[upper.tri(init_spec$init_chol_dummy, diag = FALSE)]
      init_gibbs <- TRUE
    } else {
      init_coef <- 1L
      init_coef_dummy <- 1L
      init_chol_diag <- 1L
      init_chol_upper <- 1L
      init_chol_dummy <- 1L
      init_gibbs <- FALSE
    }
    # MCMC-----------------------------
    ssvs_res <- estimate_bvar_ssvs(
      num_iter = num_iter,
      num_burn = num_burn,
      x = X1,
      y = Y0,
      init_coef = init_coef, # initial phi
      init_chol_diag = init_chol_diag, # initial psi_jj
      init_chol_upper = init_chol_upper, # initial psi_ij
      init_coef_dummy = init_coef_dummy, # initial gamma
      init_chol_dummy = init_chol_dummy, # initial omega
      coef_spike = bayes_spec$coef_spike, # phi spike
      coef_slab = bayes_spec$coef_slab, # phi slab
      coef_slab_weight = bayes_spec$coef_mixture, # pj
      shape = bayes_spec$shape, # shape of gamma distn
      rate = bayes_spec$rate, # rate of gamma distn
      coef_s1 = 1,
      coef_s2 = 1,
      chol_spike = bayes_spec$chol_spike, # eta spike
      chol_slab = bayes_spec$chol_slab, # eta slab
      chol_slab_weight = bayes_spec$chol_mixture, # qij
      chol_s1 = 1,
      chol_s2 = 1,
      grp_id = grp_id,
      grp_mat = glob_idmat,
      mean_non = bayes_spec$mean_non,
      sd_non = bayes_spec$sd_non, # c for constant c I,
      include_mean = include_mean,
      init_gibbs = init_gibbs,
      display_progress = verbose
    )
  } else {
    if (!(nrow(init_spec$init_coef[[1]]) == dim_har || ncol(init_spec$init_coef[[1]]) == dim_data)) {
      stop("Dimension of 'init_coef' should be (3 * dim) x dim or (3 * dim + 1) x dim.")
    }
    if (!(nrow(init_spec$init_coef_dummy[[1]]) == num_restrict || ncol(init_spec$init_coef_dummy[[1]]) == dim_data)) {
      stop("Dimension of 'init_coef_dummy' should be (3 * dim) x dim x dim.")
    }
    init_coef <- init_spec$init_coef
    init_coef_dummy <- init_spec$init_coef_dummy
    init_chol_diag <- lapply(init_spec$init_chol, diag)
    init_chol_upper <- lapply(
      init_spec$init_chol,
      function(x) x[upper.tri(x, diag = FALSE)]
    )
    init_chol_dummy <- lapply(
      init_spec$init_chol_dummy,
      function(x) x[upper.tri(x, diag = FALSE)]
    )
    ssvs_res <- foreach(id = seq_along(init_coef)) %dorng% {
      estimate_bvar_ssvs(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X0,
        y = Y0,
        init_coef = init_coef[[id]], # initial phi
        init_chol_diag = init_chol_diag[[id]], # initial psi_jj
        init_chol_upper = init_chol_upper[[id]], # initial psi_ij
        init_coef_dummy = init_coef_dummy[[id]], # initial gamma
        init_chol_dummy = init_chol_dummy[[id]], # initial omega
        coef_spike = bayes_spec$coef_spike, # alpha spike
        coef_slab = bayes_spec$coef_slab, # alpha slab
        coef_slab_weight = bayes_spec$coef_mixture, # pj
        shape = bayes_spec$shape, # shape of gamma distn
        rate = bayes_spec$rate, # rate of gamma distn
        coef_s1 = 1,
        coef_s2 = 1,
        chol_spike = bayes_spec$chol_spike, # eta spike
        chol_slab = bayes_spec$chol_slab, # eta slab
        chol_slab_weight = bayes_spec$chol_mixture, # qij
        chol_s1 = 1,
        chol_s2 = 1,
        mean_non = bayes_spec$mean_non,
        sd_non = bayes_spec$sd_non, # c for constant c I,
        include_mean = include_mean,
        init_gibbs = TRUE,
        display_progress = verbose
      )
    }
    ssvs_res$alpha_record <- do.call(cbind, ssvs_res$alpha_record)
    ssvs_res$eta_record <- do.call(cbind, ssvs_res$eta_record)
    ssvs_res$psi_record <- do.call(cbind, ssvs_res$psi_record)
    ssvs_res$omega_record <- do.call(cbind, ssvs_res$omega_record)
    ssvs_res$gamma_record <- do.call(cbind, ssvs_res$gamma_record)
    ssvs_res$chol_record <- do.call(cbind, ssvs_res$chol_record)
  }
  # preprocess the results------------
  names(ssvs_res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(ssvs_res))
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  ssvs_res$phi_record <- ssvs_res$phi_record[thin_id,]
  ssvs_res$eta_record <- ssvs_res$eta_record[thin_id,]
  ssvs_res$psi_record <- ssvs_res$psi_record[thin_id,]
  ssvs_res$omega_record <- ssvs_res$omega_record[thin_id,]
  ssvs_res$gamma_record <- ssvs_res$gamma_record[thin_id,]
  ssvs_res$coefficients <- colMeans(ssvs_res$phi_record)
  ssvs_res$omega_posterior <- colMeans(ssvs_res$omega_record)
  ssvs_res$pip <- colMeans(ssvs_res$gamma_record)
  if (init_spec$chain > 1) {
    ssvs_res$phi_record <- 
      split_paramarray(ssvs_res$phi_record, chain = init_spec$chain, param_name = "phi") %>% 
      as_draws_df()
    ssvs_res$gamma_record <- 
      split_paramarray(ssvs_res$gamma_record, chain = init_spec$chain, param_name = "gamma") %>% 
      as_draws_df()
    ssvs_res$psi_record <- 
      split_paramarray(ssvs_res$psi_record, chain = init_spec$chain, param_name = "psi") %>% 
      as_draws_df()
    ssvs_res$eta_record <- 
      split_paramarray(ssvs_res$eta_record, chain = init_spec$chain, param_name = "eta") %>% 
      as_draws_df()
    ssvs_res$omega_record <- 
      split_paramarray(ssvs_res$omega_record, chain = init_spec$chain, param_name = "omega") %>% 
      as_draws_df()
    ssvs_res$param <- bind_draws(
      ssvs_res$phi_record, 
      ssvs_res$gamma_record,
      ssvs_res$psi_record,
      ssvs_res$eta_record,
      ssvs_res$omega_record
    )
    # Cholesky factor 3d array-------------
    ssvs_res$chol_record <- split_psirecord(ssvs_res$chol_record, init_spec$chain, "cholesky")
    ssvs_res$chol_record <- ssvs_res$chol_record[(num_burn + 1):num_iter] # burn in
    # Posterior mean-------------------------
    ssvs_res$coefficients <- array(ssvs_res$coefficients, dim = c(dim_har, dim_data, init_spec$chain))
    # mat_upper <- array(0L, dim = c(dim_data, dim_data, ssvs_res$chain))
    
    
  } else {
    colnames(ssvs_res$phi_record) <- paste0("phi[", seq_len(ncol(ssvs_res$phi_record)), "]")
    colnames(ssvs_res$gamma_record) <- paste0("gamma[", 1:num_restrict, "]")
    colnames(ssvs_res$psi_record) <- paste0("psi[", 1:dim_data, "]")
    colnames(ssvs_res$eta_record) <- paste0("eta[", 1:num_eta, "]")
    colnames(ssvs_res$omega_record) <- paste0("omega[", 1:num_eta, "]")
    ssvs_res$phi_record <- as_draws_df(ssvs_res$phi_record)
    ssvs_res$gamma_record <- as_draws_df(ssvs_res$gamma_record)
    ssvs_res$psi_record <- as_draws_df(ssvs_res$psi_record)
    ssvs_res$eta_record <- as_draws_df(ssvs_res$eta_record)
    ssvs_res$omega_record <- as_draws_df(ssvs_res$omega_record)
    ssvs_res$param <- bind_draws(
      ssvs_res$phi_record,
      ssvs_res$gamma_record,
      ssvs_res$psi_record,
      ssvs_res$eta_record,
      ssvs_res$omega_record
    )
    # Cholesky factor 3d array---------------
    ssvs_res$chol_record <- split_psirecord(ssvs_res$chol_record, 1, "cholesky")
    ssvs_res$chol_record <- ssvs_res$chol_record[thin_id] # burn in
    # Posterior mean-------------------------
    ssvs_res$coefficients <- matrix(ssvs_res$coefficients, ncol = dim_data)
    mat_upper <- matrix(0L, nrow = dim_data, ncol = dim_data)
    diag(mat_upper) <- rep(1L, dim_data)
    mat_upper[upper.tri(mat_upper, diag = FALSE)] <- ssvs_res$omega_posterior
    ssvs_res$omega_posterior <- mat_upper
    ssvs_res$pip <- matrix(ssvs_res$pip, ncol = dim_data)
    if (include_mean) {
      ssvs_res$pip <- rbind(ssvs_res$pip, rep(1L, dim_data))
    }
    ssvs_res$chol_posterior <- Reduce("+", ssvs_res$chol_record) / length(ssvs_res$chol_record)
    # names of posterior mean-----------------
    colnames(ssvs_res$coefficients) <- name_var
    rownames(ssvs_res$coefficients) <- name_har
    colnames(ssvs_res$omega_posterior) <- name_var
    rownames(ssvs_res$omega_posterior) <- name_var
    colnames(ssvs_res$pip) <- name_var
    rownames(ssvs_res$pip) <- name_har
    colnames(ssvs_res$chol_posterior) <- name_var
    rownames(ssvs_res$chol_posterior) <- name_var
    ssvs_res$covmat <- solve(ssvs_res$chol_posterior %*% t(ssvs_res$chol_posterior))
  }
  # variables------------
  ssvs_res$df <- dim_har
  ssvs_res$p <- 3
  ssvs_res$week <- har[1]
  ssvs_res$month <- har[2]
  ssvs_res$m <- dim_data
  ssvs_res$obs <- nrow(Y0)
  ssvs_res$totobs <- nrow(y)
  # model-----------------
  ssvs_res$call <- match.call()
  ssvs_res$process <- paste("VHAR", bayes_spec$prior, sep = "_")
  ssvs_res$type <- ifelse(include_mean, "const", "none")
  ssvs_res$spec <- bayes_spec
  ssvs_res$init <- init_spec
  if (!init_gibbs) {
    ssvs_res$init$init_coef <- ssvs_res$ols_coef
    ssvs_res$init$init_coef_dummy <- matrix(1L, nrow = dim_har, ncol = dim_data)
    ssvs_res$init$init_chol <- ssvs_res$ols_cholesky
    ssvs_res$init$init_chol_dummy <- matrix(0L, nrow = dim_data, ncol = dim_data)
    ssvs_res$init$init_chol_dummy[upper.tri(ssvs_res$init$init_chol_dummy, diag = FALSE)] <- rep(1L, num_eta)
  }
  ssvs_res$iter <- num_iter
  ssvs_res$burn <- num_burn
  ssvs_res$thin <- thinning
  ssvs_res$chain <- init_spec$chain
  ssvs_res$group <- glob_idmat
  ssvs_res$num_group <- length(grp_id)
  # data------------------
  ssvs_res$HARtrans <- hartrans_mat
  ssvs_res$y0 <- Y0
  ssvs_res$design <- X0
  ssvs_res$y <- y
  # return S3 object------
  class(ssvs_res) <- c("bvharssvs", "ssvsmod", "bvharsp")
  ssvs_res
}
