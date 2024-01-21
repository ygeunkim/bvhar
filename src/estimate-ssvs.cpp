#include "mcmcssvs.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' BVAR(p) SSVS by Gibbs Sampler
//' 
//' This function conducts Gibbs sampling for BVAR SSVS.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_coef Initial k x m coefficient matrix.
//' @param init_chol_diag Inital diagonal cholesky factor
//' @param init_chol_upper Inital upper cholesky factor
//' @param init_coef_dummy Initial indicator vector (0-1) corresponding to each coefficient vector
//' @param init_chol_dummy Initial indicator vector (0-1) corresponding to each upper cholesky factor vector
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_slab_weight Coefficients vector sparsity proportion
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param coef_s1 First shape of prior beta distribution of coefficients slab weight
//' @param coef_s2 Second shape of prior beta distribution of coefficients slab weight
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Cholesky factor sparsity proportion
//' @param chol_s1 First shape of prior beta distribution of cholesky factor slab weight
//' @param chol_s2 Second shape of prior beta distribution of cholesky factor slab weight
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param mean_non Prior mean of unrestricted coefficients
//' @param sd_non Standard deviance for unrestricted coefficients
//' @param include_mean Add constant term
//' @param init_gibbs Set custom initial values for Gibbs sampler
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_chains, int num_iter, int num_burn, int thin,
                              Eigen::MatrixXd x, Eigen::MatrixXd y, 
                              Eigen::VectorXd init_coef,
                              Eigen::VectorXd init_chol_diag, Eigen::VectorXd init_chol_upper,
                              Eigen::VectorXd init_coef_dummy, Eigen::VectorXd init_chol_dummy,
                              Eigen::VectorXd coef_spike, Eigen::VectorXd coef_slab, Eigen::VectorXd coef_slab_weight,
                              Eigen::VectorXd shape, Eigen::VectorXd rate,
                              double coef_s1, double coef_s2,
                              Eigen::VectorXd chol_spike, Eigen::VectorXd chol_slab, Eigen::VectorXd chol_slab_weight,
                              double chol_s1, double chol_s2,
                              Eigen::VectorXi grp_id,
                              Eigen::MatrixXi grp_mat,
                              Eigen::VectorXd mean_non, double sd_non,
                              bool include_mean,
															Eigen::VectorXi seed_chain,
                              bool init_gibbs,
                              bool display_progress, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	std::vector<std::unique_ptr<McmcSsvs>> mcmc_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	for (int i = 0; i < num_chains; i++) {
		mcmc_objs[i] = std::unique_ptr<McmcSsvs>(new McmcSsvs(
			num_iter, x, y,
			init_coef, init_chol_diag, init_chol_upper,
			init_coef_dummy, init_chol_dummy,
			coef_spike, coef_slab, coef_slab_weight,
			shape, rate,
			coef_s1, coef_s2,
			chol_spike, chol_slab, chol_slab_weight,
			chol_s1, chol_s2,
			grp_id, grp_mat,
			mean_non, sd_non, include_mean, init_gibbs,
			static_cast<unsigned int>(seed_chain[i])
		));
	}
	// Start Gibbs sampling-----------------------------------------
	auto run_gibbs = [&](int chain) {
		bvharprogress bar(num_iter, display_progress);
		bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvharinterrupt::is_interrupted()) {
				res[chain] = mcmc_objs[chain]->returnRecords(0, 1);
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			mcmc_objs[chain]->addStep();
			mcmc_objs[chain]->doPosteriorDraws(); // Psi -> eta -> omega -> alpha -> gamma -> p
		}
		res[chain] = mcmc_objs[chain]->returnRecords(num_burn, thin);
	};
	if (num_chains == 1) {
		run_gibbs(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_gibbs(chain);
		}
	}
	return Rcpp::wrap(res);
	// std::unique_ptr<McmcSsvs> mcmc_obj(new McmcSsvs(
	// 	num_iter, x, y,
	// 	init_coef, init_chol_diag, init_chol_upper,
	// 	init_coef_dummy, init_chol_dummy,
	// 	coef_spike, coef_slab, coef_slab_weight,
	// 	shape, rate,
	// 	coef_s1, coef_s2,
	// 	chol_spike, chol_slab, chol_slab_weight,
	// 	chol_s1, chol_s2,
	// 	grp_id, grp_mat,
	// 	mean_non, sd_non, include_mean, init_gibbs
	// ));
  // bvharprogress bar(num_iter, display_progress);
	// bvharinterrupt();
  // Start Gibbs sampling-----------------------------------------
  // for (int i = 1; i < num_iter + 1; i++) {
  //   if (bvharinterrupt::is_interrupted()) {
	// 		return mcmc_obj->returnRecords(0);
  //   }
  //   bar.increment();
	// 	if (display_progress) {
	// 		bar.update();
	// 	}
	// 	mcmc_obj->addStep();
	// 	mcmc_obj->doPosteriorDraws(); // Psi -> eta -> omega -> alpha -> gamma -> p
  // }
	// return mcmc_obj->returnRecords(num_burn);
}
