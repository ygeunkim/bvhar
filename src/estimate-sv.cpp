#include "mcmcsv.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' VAR-SV by Gibbs Sampler
//' 
//' This function generates parameters \eqn{\beta, a, \sigma_{h,i}^2, h_{0,i}} and log-volatilities \eqn{h_{i,1}, \ldots, h_{i, n}}.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param prior_coef_mean Prior mean matrix of coefficient in Minnesota belief
//' @param prior_coef_prec Prior precision matrix of coefficient in Minnesota belief
//' @param prec_diag Diagonal matrix of sigma of innovation to build Minnesota moment
//' @param init_local Initial local shrinkage of Horseshoe
//' @param init_global Initial global shrinkage of Horseshoe
//' @param init_contem_local Initial local shrinkage for Cholesky factor in Horseshoe
//' @param init_contem_global Initial global shrinkage for Cholesky factor in Horseshoe
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param prior_sig_shp Inverse-Gamma prior shape of state variance
//' @param prior_sig_scl Inverse-Gamma prior scale of state variance
//' @param prior_init_mean Noraml prior mean of initial state
//' @param prior_init_prec Normal prior precision of initial state
//' @param coef_spike SD of spike normal
//' @param coef_slab_weight SD of slab normal
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Cholesky factor sparsity proportion
//' @param coef_s1 First shape of prior beta distribution of coefficients slab weight
//' @param coef_s2 Second shape of prior beta distribution of coefficients slab weight
//' @param mean_non Prior mean of unrestricted coefficients
//' @param sd_non SD for unrestricted coefficients
//' @param include_mean Constant term
//' @param display_progress Progress bar
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var_sv(int num_iter, int num_burn,
                           Eigen::MatrixXd x, Eigen::MatrixXd y,
													 Rcpp::List param_sv,
													 Rcpp::List param_prior,
                           int prior_type,
                           Eigen::VectorXi grp_id,
                           Eigen::MatrixXd grp_mat,
                           bool include_mean,
                           bool display_progress, int nthreads) {
  std::unique_ptr<McmcSv> sv_obj;
	switch (prior_type) {
		case 1: {
			MinnParams minn_params(
				num_iter, x, y,
				param_sv, param_prior
			);
			sv_obj = std::unique_ptr<McmcSv>(new MinnSv(minn_params));
			break;
		}
		case 2: {
			SsvsParams ssvs_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_prior,
				include_mean
			);
			sv_obj = std::unique_ptr<McmcSv>(new SsvsSv(ssvs_params));
			break;
		}
		case 3: {
			HorseshoeParams horseshoe_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_prior
			);
			sv_obj = std::unique_ptr<McmcSv>(new HorseshoeSv(horseshoe_params));
			break;
		}
	}
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
  Eigen::initParallel();
#endif
  // Start Gibbs sampling-----------------------------------
	bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
  for (int i = 0; i < num_iter; i++) {
		if (bvharinterrupt::is_interrupted()) {
			return sv_obj->returnRecords(num_burn);
    }
		bar.increment();
		if (display_progress) {
			bar.update();
		}
		sv_obj->addStep();
		sv_obj->doPosteriorDraws(); // a -> alpha -> h -> sigma_h -> h0
  }
	return sv_obj->returnRecords(num_burn);
}
