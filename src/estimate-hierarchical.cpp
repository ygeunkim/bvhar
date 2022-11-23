#ifdef _OPENMP
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

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
double jointdens_hyperparam(double cand_gamma,
                            Eigen::VectorXd cand_invgam,
                            int dim, 
                            int num_design,
                            Eigen::MatrixXd prior_prec,
                            Eigen::MatrixXd prior_scale,
                            int prior_shape,
                            Eigen::MatrixXd mn_prec,
                            Eigen::MatrixXd iw_scale,
                            int posterior_shape,
                            int gamma_shp,
                            int gamma_rate,
                            int invgam_shp,
                            int invgam_scl) {
  double res = compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_shape);
  res += -dim * num_design / 2.0 * log(M_PI) +
    log_mgammafn((prior_shape + num_design) / 2.0, dim) -
    log_mgammafn(prior_shape / 2.0, dim); // constant term
  // for (int i = 0; i < cand_gamma.size(); i++) {
  //   res += gamma_dens(cand_gamma[i], gamma_shp, 1 / gamma_rate, true); // gamma distribution
  // }
  res += gamma_dens(cand_gamma, gamma_shp, 1 / gamma_rate, true);
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
//' @param num_burn Number of burn-in for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param obs_information Observed Fisher information matrix
//' @param acc_scale Scaling constant for acceptance rate to be about 20 percent
//' @param init_lambda Initial lambda
//' @param init_psi 
//' @param init_coef
//' @param init_sig
//' @param chain The number of MCMC chains
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_hierachical_niw(int num_iter,
                                    int num_burn,
                                    Eigen::MatrixXd x, 
                                    Eigen::MatrixXd y,
                                    Eigen::MatrixXd prior_prec,
                                    Eigen::MatrixXd prior_scale,
                                    int prior_shape,
                                    Eigen::MatrixXd mn_prec,
                                    Eigen::MatrixXd iw_scale,
                                    int posterior_shape,
                                    int gamma_shp,
                                    int gamma_rate,
                                    int invgam_shp,
                                    int invgam_scl,
                                    Eigen::MatrixXd obs_information,
                                    double acc_scale,
                                    Eigen::VectorXd init_lambda,
                                    Eigen::VectorXd init_psi,
                                    Eigen::MatrixXd init_coef,
                                    Eigen::MatrixXd init_sig,
                                    int chain) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  Eigen::MatrixXd gaussian_variance = acc_scale * obs_information.inverse();
  // Initialize coefficients vector-------------------------------
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  Eigen::MatrixXd coef_ols = XtX.inverse() * x.transpose() * y;
  // Bayes point estimates----------------------------------------
  
  // record-------------------------------------------------------
  Eigen::VectorXd lam_record = Eigen::MatrixXd::Zero(num_iter, chain);
  lam_record.row(0) = init_lambda;
  Eigen::MatrixXd psi_record = Eigen::MatrixXd::Zero(num_iter, dim * chain);
  psi_record.row(0) = init_psi;
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(dim_design * num_iter, dim * chain);
  coef_record.topRows(dim_design) = init_coef;
  Eigen::MatrixXd sig_record = Eigen::MatrixXd::Zero(dim * num_iter, dim * chain);
  sig_record.topRows(dim) = init_sig;
  // Some variables-----------------------------------------------
  Eigen::VectorXd prevprior = Eigen::VectorXd::Zero((1 + dim) * chain);
  prevprior.segment(0, chain) = init_lambda;
  prevprior.segment(chain, dim * chain) = init_psi;
  Eigen::VectorXd candprior = Eigen::VectorXd::Zero((1 + dim) * chain);
  Rcpp::List posterior_draw;
  double numerator = 0;
  double denom = 0;
  // Start Metropolis---------------------------------------------
  typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
  VectorXb is_accept(num_iter + 1);
  is_accept[0] = true;
  for (int i = 1; i < num_iter; i ++) {
    candprior = sim_mgaussian_chol(1, prevprior, gaussian_variance);
    numerator = jointdens_hyperparam(candprior[0], candprior.segment(1, dim), dim, num_design, prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl);
    denom = jointdens_hyperparam(prevprior[0], prevprior.segment(1, dim), dim, num_design, prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl);
    is_accept[i] = ( log(unif_rand(0, 1)) < std::min(numerator - denom, 0.0) );
    if (is_accept[i]) {
      lam_record.row(i) = candprior.segment(0, 1);
      psi_record.row(i) = candprior.segment(1, dim);
    } else {
      lam_record.row(i) = lam_record.row(i - 1);
      psi_record.row(i) = psi_record.row(i - 1);
    }
    // Draw coef and Sigma
    
  }
  
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record,
    Rcpp::Named("sigma_record") = sig_record,
    Rcpp::Named("coefficients") = coef_ols,
    Rcpp::Named("chain") = chain
  );
}
