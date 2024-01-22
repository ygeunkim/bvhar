#include <mcmcsv.h>
#include <bvharinterrupt.h>

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
Rcpp::List estimate_var_sv(int num_chains, int num_iter, int num_burn, int thin,
                           Eigen::MatrixXd x, Eigen::MatrixXd y,
													 Rcpp::List param_sv,
													 Rcpp::List param_prior,
													 Rcpp::List param_intercept,
													 Rcpp::List param_init,
                           int prior_type,
                           Eigen::VectorXi grp_id,
                           Eigen::MatrixXi grp_mat,
                           bool include_mean,
													 Eigen::VectorXi seed_chain,
                           bool display_progress, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	std::vector<std::unique_ptr<McmcSv>> sv_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	switch (prior_type) {
		case 1: {
			MinnParams minn_params(
				num_iter, x, y,
				param_sv, param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				SvInits sv_inits(init_spec);
				sv_objs[i] = std::unique_ptr<McmcSv>(new MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 2: {
			SsvsParams ssvs_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				SsvsInits ssvs_inits(init_spec);
				sv_objs[i] = std::unique_ptr<McmcSv>(new SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 3: {
			HorseshoeParams horseshoe_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				HorseshoeInits hs_inits(init_spec);
				sv_objs[i] = std::unique_ptr<McmcSv>(new HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
	}
  // Start Gibbs sampling-----------------------------------
	auto run_gibbs = [&](int chain) {
		bvharprogress bar(num_iter, display_progress);
		bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvharinterrupt::is_interrupted()) {
				res[chain] = sv_objs[chain]->returnRecords(0, 1);
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			sv_objs[chain]->addStep();
			sv_objs[chain]->doPosteriorDraws(); // alpha -> a -> h -> sigma_h -> h0
		}
		res[chain] = sv_objs[chain]->returnRecords(num_burn, thin);
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
}
