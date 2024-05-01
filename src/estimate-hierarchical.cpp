#include "bvhardraw.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' Metropolis Algorithm for Normal-IW Hierarchical Model
//' 
//' This function conducts Metropolis algorithm for Normal-IW Hierarchical BVAR or BVHAR.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param prior_prec Prior precision of Matrix Normal distribution
//' @param prior_scale Prior scale of Inverse-Wishart distribution
//' @param prior_shape Prior degrees of freedom of Inverse-Wishart distribution
//' @param mn_mean Posterior mean of Matrix Normal distribution
//' @param mn_prec Posterior precision of Matrix Normal distribution
//' @param iw_scale Posterior scale of Inverse-Wishart distribution
//' @param posterior_shape Posterior degrees of freedom of Inverse-Wishart distribution
//' @param gamma_shp Shape of hyperprior Gamma distribution
//' @param gamma_rate Rate of hyperprior Gamma distribution
//' @param invgam_shp Shape of hyperprior Inverse gamma distribution
//' @param invgam_scl Scale of hyperprior Inverse gamma distribution
//' @param acc_scale Proposal distribution scaling constant to adjust an acceptance rate
//' @param obs_information Observed Fisher information matrix
//' @param init_lambda Initial lambda
//' @param init_psi Initial psi
//' @param init_coef Initial coefficients
//' @param init_sig Initial sig
//' @param display_progress Progress bar
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_hierachical_niw(int num_iter, int num_burn, Eigen::MatrixXd x, Eigen::MatrixXd y,
                                    Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, int prior_shape,
                                    Eigen::MatrixXd mn_mean, Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale,
                                    int posterior_shape, double gamma_shp, double gamma_rate, double invgam_shp, double invgam_scl, double acc_scale,
                                    Eigen::MatrixXd obs_information, double init_lambda, Eigen::VectorXd init_psi,
                                    bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  Eigen::MatrixXd gaussian_variance = acc_scale * obs_information.inverse();
  // record-------------------------------------------------------
  Eigen::VectorXd lam_record(num_iter + 1);
  lam_record[0] = init_lambda;
  Eigen::MatrixXd psi_record(num_iter + 1, dim);
  psi_record.row(0) = init_psi;
  Eigen::MatrixXd coef_record(num_iter, dim * dim_design);
  Eigen::MatrixXd sig_record(dim * num_iter, dim);
  // Some variables-----------------------------------------------
  Eigen::VectorXd prevprior = Eigen::VectorXd::Zero(1 + dim);
  // prevprior.segment(0, chain) = init_lambda;
  prevprior[0] = init_lambda;
  prevprior.segment(1, dim) = init_psi;
  Eigen::VectorXd candprior = Eigen::VectorXd::Zero(1 + dim);
  // Rcpp::List posterior_draw = Rcpp::List::create(
  //   Rcpp::Named("mn") = Eigen::MatrixXd::Zero(dim_design, dim),
  //   Rcpp::Named("iw") = Eigen::MatrixXd::Zero(dim, dim)
  // );
	std::vector<Eigen::MatrixXd> posterior_draw(2);
  double numerator = 0;
  double denom = 0;
  bvhar::bvharprogress bar(num_iter, display_progress);
	bvhar::bvharinterrupt();
  // Start Metropolis---------------------------------------------
  bvhar::VectorXb is_accept(num_iter + 1);
  is_accept[0] = true;
  for (int i = 1; i < num_iter + 1; i ++) {
    if (bvhar::bvharinterrupt::is_interrupted()) {
      return Rcpp::List::create(
        Rcpp::Named("lambda_record") = lam_record,
        Rcpp::Named("psi_record") = psi_record,
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("sigma_record") = sig_record,
        Rcpp::Named("acceptance") = is_accept
      );
    }
    bar.increment();
		if (display_progress) {
			bar.update();
		}
    // Candidate ~ N(previous, scaled hessian)
    candprior = Eigen::Map<Eigen::VectorXd>(sim_mgaussian_chol(1, prevprior, gaussian_variance).data(), 1 + dim);
    // log of acceptance rate = numerator - denom
    numerator = bvhar::jointdens_hyperparam(
			candprior[0], candprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
    denom = bvhar::jointdens_hyperparam(
			prevprior[0], prevprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
    is_accept[i] = ( log(bvhar::unif_rand(0, 1)) < std::min(numerator - denom, 0.0) );
    // Update
    if (is_accept[i]) {
      lam_record[i] = candprior[0];
      psi_record.row(i) = candprior.segment(1, dim);
    } else {
      lam_record[i] = lam_record[i - 1];
      psi_record.row(i) = psi_record.row(i - 1);
    }
    // Draw coef and Sigma
    // posterior_draw = sim_mniw(1, mn_mean, mn_prec.inverse(), iw_scale, posterior_shape);
		// coef_record.row(i - 1) = bvhar::vectorize_eigen(Rcpp::as<Eigen::MatrixXd>(posterior_draw["mn"]));
    // sig_record.block((i - 1) * dim, 0, dim, dim) = Rcpp::as<Eigen::MatrixXd>(posterior_draw["iw"]);
		posterior_draw = bvhar::sim_mn_iw(mn_mean, mn_prec.inverse(), iw_scale, posterior_shape, false);
		coef_record.row(i - 1) = bvhar::vectorize_eigen(posterior_draw[0]);
    sig_record.block((i - 1) * dim, 0, dim, dim) = posterior_draw[1];
  }
  return Rcpp::List::create(
    Rcpp::Named("lambda_record") = lam_record.tail(num_iter - num_burn),
    Rcpp::Named("psi_record") = psi_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sigma_record") = sig_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("acceptance") = is_accept.tail(num_iter - num_burn)
  );
}
