#ifdef _OPENMP
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

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
                            double gamma_shp,
                            double gamma_rate,
                            double invgam_shp,
                            double invgam_scl) {
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
//' @param num_warm Number of warm-up (burn-in) for MCMC
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
//' @param chain The number of MCMC chains
//' @param display_progress Progress bar
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_hierachical_niw(int num_iter,
                                    int num_warm,
                                    Eigen::MatrixXd x, 
                                    Eigen::MatrixXd y,
                                    Eigen::MatrixXd prior_prec,
                                    Eigen::MatrixXd prior_scale,
                                    int prior_shape,
                                    Eigen::MatrixXd mn_mean,
                                    Eigen::MatrixXd mn_prec,
                                    Eigen::MatrixXd iw_scale,
                                    int posterior_shape,
                                    double gamma_shp,
                                    double gamma_rate,
                                    double invgam_shp,
                                    double invgam_scl,
                                    double acc_scale,
                                    Eigen::MatrixXd obs_information,
                                    Eigen::VectorXd init_lambda,
                                    Eigen::VectorXd init_psi,
                                    int chain,
                                    bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  Eigen::MatrixXd gaussian_variance = acc_scale * obs_information.inverse();
  // record-------------------------------------------------------
  Eigen::VectorXd lam_record = Eigen::MatrixXd::Zero(num_iter, chain);
  lam_record.row(0) = init_lambda;
  Eigen::MatrixXd psi_record = Eigen::MatrixXd::Zero(num_iter, dim * chain);
  psi_record.row(0) = init_psi;
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter - 1, dim * dim_design * chain);
  Eigen::MatrixXd sig_record = Eigen::MatrixXd::Zero(dim * (num_iter - 1), dim * chain);
  // Some variables-----------------------------------------------
  Eigen::VectorXd prevprior = Eigen::VectorXd::Zero((1 + dim) * chain);
  prevprior.segment(0, chain) = init_lambda;
  prevprior.segment(chain, dim * chain) = init_psi;
  Eigen::VectorXd candprior = Eigen::VectorXd::Zero((1 + dim) * chain);
  Rcpp::List posterior_draw = Rcpp::List::create(
    Rcpp::Named("mn") = Eigen::MatrixXd::Zero(dim_design, dim),
    Rcpp::Named("iw") = Eigen::MatrixXd::Zero(dim, dim)
  );
  double numerator = 0;
  double denom = 0;
  Progress p(chain * (num_iter - 1), display_progress);
  // Start Metropolis---------------------------------------------
  typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
  VectorXb is_accept(num_iter + 1);
  is_accept[0] = true;
#ifdef _OPENMP
  Rcpp::Rcout << "Use parallel" << std::endl;
#pragma                  \
  omp                    \
    parallel             \
    for                  \
      num_threads(chain) \
      shared(gaussian_variance, mn_mean, mn_prec, iw_scale, posterior_shape, dim_design, dim)
  for (int b = 0; b < chain; b++) {
    for (int i = 1; i < num_iter; i ++) {
      candprior = Eigen::Map<Eigen::VectorXd>(sim_mgaussian_chol(1, prevprior, gaussian_variance).data(), 1 + dim);
      numerator = jointdens_hyperparam(candprior[0], candprior.segment(1, dim), dim, num_design, prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl);
      denom = jointdens_hyperparam(prevprior[0], prevprior.segment(1, dim), dim, num_design, prior_prec, prior_scale, prior_shape, mn_prec, iw_scale, posterior_shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl);
      is_accept[i] = ( log(unif_rand(0, 1)) < std::min(numerator - denom, 0.0) );
      if (is_accept[i]) {
        lam_record.block(i, b * dim, 1, 1) = candprior.segment(b * (dim + 1), 1);
        psi_record.block(i, b * dim, 1, dim) = candprior.segment(b * (dim + 1) + 1, dim);
      } else {
        lam_record.block(i, b * dim, 1, 1) = lam_record.block(i - 1, b * dim, 1, 1);
        psi_record.block(i, b * dim, 1, dim) = psi_record.block(i - 1, b * dim, 1, dim);
      }
      posterior_draw = sim_mniw(
        1,
        mn_mean,
        mn_prec.inverse(),
        iw_scale,
        posterior_shape
      );
      // coef_record.block(i * dim_design, b * dim, dim_design, dim) = Eigen::Map<Eigen::MatrixXd>(posterior_draw["mn"], dim_design, dim);
      coef_record.block(i - 1, b * dim, 1, dim * dim_design) = vectorize_eigen(posterior_draw["mn"]);
      sig_record.block((i - 1) * dim, b * dim, dim, dim) = Rcpp::as<Eigen::MatrixXd>(posterior_draw["iw"]);
    }
  }
#else
  for (int i = 1; i < num_iter; i ++) {
    if (Progress::check_abort()) {
      return Rcpp::List::create(
        Rcpp::Named("lambda_record") = lam_record,
        Rcpp::Named("psi_record") = psi_record,
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("sigma_record") = sig_record,
        Rcpp::Named("acceptance") = is_accept,
        Rcpp::Named("chain") = chain
      );
    }
    p.increment();
    // Candidate ~ N(previous, scaled hessian)
    candprior = Eigen::Map<Eigen::VectorXd>(sim_mgaussian_chol(1, prevprior, gaussian_variance).data(), 1 + dim);
    // log of acceptance rate = numerator - denom
    numerator = jointdens_hyperparam(
      candprior[0],
      candprior.segment(1, dim),
      dim,
      num_design,
      prior_prec,
      prior_scale,
      prior_shape,
      mn_prec,
      iw_scale,
      posterior_shape,
      gamma_shp,
      gamma_rate,
      invgam_shp,
      invgam_scl
    );
    denom = jointdens_hyperparam(
      prevprior[0],
      prevprior.segment(1, dim),
      dim,
      num_design,
      prior_prec,
      prior_scale,
      prior_shape,
      mn_prec,
      iw_scale,
      posterior_shape,
      gamma_shp,
      gamma_rate,
      invgam_shp,
      invgam_scl
    );
    is_accept[i] = ( log(unif_rand(0, 1)) < std::min(numerator - denom, 0.0) );
    // Update
    if (is_accept[i]) {
      lam_record.row(i) = candprior.segment(0, 1);
      psi_record.row(i) = candprior.segment(1, dim);
    } else {
      lam_record.row(i) = lam_record.row(i - 1);
      psi_record.row(i) = psi_record.row(i - 1);
    }
    // Draw coef and Sigma
    posterior_draw = sim_mniw(
      1,
      mn_mean,
      mn_prec.inverse(),
      iw_scale,
      posterior_shape
    );
    coef_record.row(i - 1) = vectorize_eigen(posterior_draw["mn"]);
    sig_record.block((i - 1) * dim, 0, dim, dim) = Rcpp::as<Eigen::MatrixXd>(posterior_draw["iw"]);
  }
#endif
  return Rcpp::List::create(
    Rcpp::Named("lambda_record") = lam_record.bottomRows(num_iter - num_warm),
    Rcpp::Named("psi_record") = psi_record.bottomRows(num_iter - num_warm),
    Rcpp::Named("alpha_record") = coef_record.bottomRows((num_iter - 1) - num_warm),
    Rcpp::Named("sigma_record") = sig_record,
    Rcpp::Named("acceptance") = is_accept.tail(num_iter - num_warm),
    Rcpp::Named("chain") = chain
  );
}
