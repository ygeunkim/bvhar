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
                           Eigen::MatrixXd prior_coef_mean,
                           Eigen::MatrixXd prior_coef_prec,
                           Eigen::MatrixXd prec_diag,
                           int prior_type,
                           Eigen::VectorXd init_local,
                           Eigen::VectorXd init_global,
                           Eigen::VectorXd init_contem_local,
                           Eigen::VectorXd init_contem_global,
                           Eigen::VectorXi grp_id,
                           Eigen::MatrixXd grp_mat,
													 Eigen::VectorXd prior_sig_shp,
													 Eigen::VectorXd prior_sig_scl,
													 Eigen::VectorXd prior_init_mean,
													 Eigen::MatrixXd prior_init_prec,
                           Eigen::VectorXd coef_spike,
                           Eigen::VectorXd coef_slab,
                           Eigen::VectorXd coef_slab_weight,
                           Eigen::VectorXd chol_spike,
                           Eigen::VectorXd chol_slab,
                           Eigen::VectorXd chol_slab_weight,
                           double coef_s1, double coef_s2,
                           double chol_s1, double chol_s2,
                           Eigen::VectorXd mean_non,
                           double sd_non,
                           bool include_mean,
                           bool display_progress, int nthreads) {
  std::unique_ptr<McmcSv> sv_obj;
	switch (prior_type) {
		case 1: {
			MinnParams minn_params(
				num_iter, x, y, prior_sig_shp, prior_sig_scl, prior_init_mean, prior_init_prec,
				prior_coef_mean, prior_coef_prec, prec_diag
			);
			sv_obj = std::unique_ptr<McmcSv>(new MinnSv(minn_params));
			break;
		}
		case 2: {
			SsvsParams ssvs_params(
				num_iter, x, y, prior_sig_shp, prior_sig_scl, prior_init_mean, prior_init_prec,
				grp_id, grp_mat, coef_spike, coef_slab, coef_slab_weight, chol_spike, chol_slab, chol_slab_weight,
				coef_s1, coef_s2, chol_s1, chol_s2,
				mean_non, sd_non, include_mean
			);
			sv_obj = std::unique_ptr<McmcSv>(new SsvsSv(ssvs_params));
			break;
		}
		case 3: {
			HorseshoeParams horseshoe_params(
				num_iter, x, y, prior_sig_shp, prior_sig_scl, prior_init_mean, prior_init_prec,
				grp_id, grp_mat, init_local, init_global, init_contem_local, init_contem_global
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
  for (int i = 1; i < num_iter + 1; i ++) {
		if (bvharinterrupt::is_interrupted()) {
			return sv_obj->returnRecords(num_burn);
    }
		bar.increment();
		if (display_progress) {
			bar.update();
		}
		sv_obj->addStep();
		sv_obj->doPosteriorDraws(); // alpha -> a -> h -> sigma_h -> h0
  }
	return sv_obj->returnRecords(num_burn);
}
