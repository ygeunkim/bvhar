// #include <RcppEigen.h>
// #include "bvharmisc.h"
// #include "randsim.h"
#include "minnesota.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' Log of Joint Posterior Density of Hyperparameters
//' 
//' This function computes the log of joint posterior density of hyperparameters.
//' 
//' @param cand_gamma Candidate value of hyperparameters following Gamma distribution
//' @param cand_invgam Candidate value of hyperparameters following Inverse Gamma distribution
//' @param dim Dimension of the time series
//' @param num_design The number of the data matrix, \eqn{n = T - p}
//' @param prior_prec Prior precision of Matrix Normal distribution
//' @param prior_scale Prior scale of Inverse-Wishart distribution
//' @param mn_prec Posterior precision of Matrix Normal distribution
//' @param iw_scale Posterior scale of Inverse-Wishart distribution
//' @param posterior_shape Posterior shape of Inverse-Wishart distribution
//' @param gamma_shape Shape of hyperprior Gamma distribution
//' @param gamma_rate Rate of hyperprior Gamma distribution
//' @param invgam_shape Shape of hyperprior Inverse gamma distribution
//' @param invgam_scl Scale of hyperprior Inverse gamma distribution
//' 
//' @noRd
// [[Rcpp::export]]
double jointdens_hyperparam(double cand_gamma, Eigen::VectorXd cand_invgam, int dim, int num_design,
                            Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, int prior_shape,
                            Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale,
                            int posterior_shape, double gamma_shp, double gamma_rate, double invgam_shp, double invgam_scl) {
  double res = compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_shape);
  res += -dim * num_design / 2.0 * log(M_PI) +
    log_mgammafn((prior_shape + num_design) / 2.0, dim) -
    log_mgammafn(prior_shape / 2.0, dim); // constant term
  res += gamma_dens(cand_gamma, gamma_shp, 1 / gamma_rate, true); // gamma distribution
  for (int i = 0; i < cand_invgam.size(); i++) {
    res += invgamma_dens(cand_invgam[i], invgam_shp, invgam_scl, true); // inverse gamma distribution
  }
  return res;
}

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
																		Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy,
																		Rcpp::List init_spec,
																		Rcpp::List hyper_spec,
                                    bool display_progress) {
	MinnSpec bayes_spec(init_spec);
	HierMinnSpec hmn_spec(hyper_spec);
	std::unique_ptr<HierMinn> hmn_obj(new HierMinn(
		num_iter, x, y, x_dummy, y_dummy, hmn_spec, bayes_spec
	));
  // Start Metropolis---------------------------------------------
	bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
	for (int i = 0; i < num_iter; i++) {
    if (bvharinterrupt::is_interrupted()) {
			return hmn_obj->returnRecords(0);
    }
    bar.increment();
		if (display_progress) {
			bar.update();
		}
		hmn_obj->addStep();
		hmn_obj->doPosteriorDraws();
  }
	return hmn_obj->returnRecords(num_burn);
}
