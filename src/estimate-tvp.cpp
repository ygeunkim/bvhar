#include <RcppEigen.h>
#include "bvhardraw.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' TVP-VAR by Gibbs Sampler
//' 
//' This function generates parameters \eqn{\beta, \beta_0, \Sigma, Q}.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param prior_coef_mean Prior mean matrix of coefficient in Minnesota belief
//' @param prior_coef_prec Prior precision matrix of coefficient in Minnesota belief
//' @param prec_diag Diagonal matrix of sigma of innovation to build Minnesota moment
//' @param prior_sig_df Prior df of Minnesota IW
//' @param prior_sig_scale Prior scale of Minnesota IW
//' @param display_progress Progress bar
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var_tvp(int num_iter, int num_burn, Eigen::MatrixXd x, Eigen::MatrixXd y,
                            Eigen::MatrixXd prior_coef_mean, Eigen::MatrixXd prior_coef_prec, Eigen::MatrixXd prec_diag,
                            double prior_sig_df, Eigen::MatrixXd prior_sig_scale,
                            bool display_progress, int nthreads) {
  int dim = y.cols(); // k
  int dim_design = x.cols(); // kp(+1)
  int num_design = y.rows(); // n = T - p
  int num_coef = dim * dim_design;
  // SUR---------------------------------------------------
  Eigen::VectorXd response_vec = vectorize_eigen(y); // y = vec(Y0)
  // Eigen::MatrixXd design_mat = kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), x);
  Eigen::MatrixXd sur_tvp = Eigen::MatrixXd::Zero(num_design * dim, num_design * num_coef); // diag(X_1, ..., X_n)
  for (int i = 0; i < num_design; i++) {
    sur_tvp.block(i * dim, i * num_coef, dim, num_coef) = kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), x.row(i));
  }
  // Default setting---------------------------------------
  Eigen::VectorXd prior_alpha_mean = vectorize_eigen(prior_coef_mean); // prior mean vector of alpha
  Eigen::MatrixXd prior_alpha_prec = kronecker_eigen(prec_diag, prior_coef_prec); // prior precision of alpha
  Eigen::VectorXd prior_coefsig_shp = 3 * Eigen::VectorXd::Ones(num_coef); // nu0 of q_i
  Eigen::VectorXd prior_coefsig_scl = .01 * Eigen::VectorXd::Ones(num_coef); // S_0 of q_i
  // record------------------------------------------------
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_design * num_coef); // time-varying coefficients (alpha_1, ... alpha_n)
  // Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_design * (num_iter + 1), num_coef); // time-varying coefficients (alpha_1, ... alpha_n)
  Eigen::MatrixXd coef_init_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef); // alpha_0
  Eigen::MatrixXd sig_record = Eigen::MatrixXd::Zero(dim * (num_iter + 1), dim); // covariance of innovation
  Eigen::MatrixXd coef_sig_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef); // Covariance time-varying alpha(random-walk) Q = diag(q1, ..., q_(k,kp(+1)))
  // Initialization----------------------------------------
  Eigen::MatrixXd coef_ols = (x.transpose() * x).llt().solve(x.transpose() * y); // LSE
  coef_init_record.row(0) = vectorize_eigen(coef_ols);
  // coef_record.row(0) = coef_init_record.row(0).replicate(1, num_design);
  // coef_record.block(0, 0, num_design, num_coef) = coef_init_record.row(0).replicate(num_design, 1);
  sig_record.block(0, 0, dim, dim) = (y - x * coef_ols).transpose() * (y - x * coef_ols) / (num_design - dim_design);
  coef_sig_record.row(0) = Eigen::VectorXd::Ones(num_coef);
  // Some variables----------------------------------------
  Eigen::MatrixXd coef_varying(num_design, num_coef); // matrix form of time-varying alpha
  // Eigen::MatrixXd coef_mat(dim_design, dim); // A
  // Eigen::VectorXd coef_vec(num_design * num_coef); // vectorized time-varying alpha
  Eigen::MatrixXd coef_prec = Eigen::MatrixXd::Zero(num_coef, num_coef); // Q^(-1)
  coef_prec.diagonal() = coef_sig_record.row(0).array();
  Eigen::MatrixXd prec_mat = Eigen::MatrixXd::Zero(dim, dim);
  prec_mat = sig_record.block(0, 0, dim, dim).inverse();
  Eigen::VectorXd yt_xalpha = Eigen::VectorXd::Zero(dim); // yt - Xt alpha_t
  Eigen::MatrixXd iw_scl = Eigen::MatrixXd::Zero(dim, dim);
  // Start Gibbs sampling-----------------------------------
  bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
  for (int i = 1; i < num_iter + 1; i ++) {
    if (bvharinterrupt::is_interrupted()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("alpha0_record") = coef_init_record,
        Rcpp::Named("sig_record") = sig_record,
        Rcpp::Named("q_record") = coef_sig_record
      );
    }
    bar.increment();
		if (display_progress) {
			bar.update();
		}
    // 1. alpha
    for (int t = 0; t < num_design; t++) {
      // coef_record.block(num_design * i, 0, num_design, num_coef).row(t) = varsv_regression(sur_tvp.block(t * dim, t * num_coef, dim, num_coef), y.row(t), coef_init_record.row(i - 1), coef_prec, prec_mat);
      // coef_record.row(i).segment(num_coef * t, num_coef) = varsv_regression(sur_tvp.block(t * dim, t * num_coef, dim, num_coef), y.row(t), coef_init_record.row(i - 1), coef_prec, prec_mat);
			coef_record.row(i).segment(num_coef * t, num_coef) = tvp_coef(sur_tvp.block(t * dim, t * num_coef, dim, num_coef), y.row(t), coef_init_record.row(i - 1), coef_prec, prec_mat);
    }
    // 2. sigma
    iw_scl = Eigen::MatrixXd::Zero(dim, dim);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) private(yt_xalpha)
    for (int t = 0; t < num_design; t++) {
      // yt_xalpha = y.row(t).transpose() - sur_tvp.block(t * dim, t * num_coef, dim, num_coef) * coef_record.block(num_design * i, 0, num_design, num_coef).row(t).transpose();
      yt_xalpha = y.row(t).transpose() - sur_tvp.block(t * dim, t * num_coef, dim, num_coef) * coef_record.row(i).segment(num_coef * t, num_coef).transpose();
      iw_scl += yt_xalpha * yt_xalpha.transpose();
    }
#else
    for (int t = 0; t < num_design; t++) {
      // yt_xalpha = y.row(t).transpose() - sur_tvp.block(t * dim, t * num_coef, dim, num_coef) * coef_record.block(num_design * i, 0, num_design, num_coef).row(t).transpose();
      yt_xalpha = y.row(t).transpose() - sur_tvp.block(t * dim, t * num_coef, dim, num_coef) * coef_record.row(i).segment(num_coef * t, num_coef).transpose();
      iw_scl += yt_xalpha * yt_xalpha.transpose();
    }
#endif
    sig_record.block(dim * i, 0, dim, dim) = sim_iw(prior_sig_scale + iw_scl, prior_sig_df + num_design);
    prec_mat = sig_record.block(dim * i, 0, dim, dim).inverse();
    // 3. Q
    // coef_sig_record.row(i) = varsv_sigh(prior_coefsig_shp, prior_coefsig_scl, coef_sig_record.row(i - 1), coef_record.block(num_design * i, 0, num_design, num_coef));
    coef_varying = Eigen::Map<Eigen::MatrixXd>(coef_record.row(i).data(), num_design, num_coef);
    coef_sig_record.row(i) = varsv_sigh(prior_coefsig_shp, prior_coefsig_scl, coef_sig_record.row(i - 1), coef_varying);
    // 4. alpha_0
    coef_prec.diagonal() = 1 / coef_sig_record.row(i).array();
    // coef_init_record.row(i) = tvp_initcoef(prior_alpha_mean, prior_alpha_prec, coef_record.block(num_design * i, 0, num_design, num_coef).row(0), coef_prec);
    coef_init_record.row(i) = tvp_initcoef(prior_alpha_mean, prior_alpha_prec, coef_record.row(i).segment(0, num_coef), coef_prec);
  }
  return Rcpp::List::create(
    // Rcpp::Named("alpha_record") = coef_record.bottomRows(num_design * (num_iter - num_burn)),
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("alpha0_record") = coef_init_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sig_record") = sig_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("q_record") = coef_sig_record.bottomRows(num_iter - num_burn)
  );
}
