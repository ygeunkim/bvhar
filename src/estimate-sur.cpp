#include "mcmcreg.h"
#include "bvharinterrupt.h"

//' VAR with Shrinkage Priors
//' 
//' This function generates parameters \eqn{\beta, a, \sigma_{h,i}^2, h_{0,i}} and log-volatilities \eqn{h_{i,1}, \ldots, h_{i, n}}.
//' 
//' @param num_chain Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thin Thinning
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param param_reg Regression specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param seed_chain Seed for each chain
//' @param display_progress Progress bar
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_sur(int num_chains, int num_iter, int num_burn, int thin,
                        Eigen::MatrixXd x, Eigen::MatrixXd y,
												Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept,
												Rcpp::List param_init, int prior_type,
                        Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
                        bool include_mean, Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	std::vector<std::unique_ptr<bvhar::McmcReg>> sur_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	switch (prior_type) {
		case 1: {
			bvhar::MinnParams minn_params(
				num_iter, x, y,
				param_reg, param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::LdltInits ldlt_inits(init_spec);
				sur_objs[i].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 2: {
			bvhar::SsvsParams ssvs_params(
				num_iter, x, y,
				param_reg,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::SsvsInits ssvs_inits(init_spec);
				sur_objs[i].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 3: {
			bvhar::HorseshoeParams horseshoe_params(
				num_iter, x, y,
				param_reg,
				grp_id, grp_mat,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::HsInits hs_inits(init_spec);
				sur_objs[i].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 4: {
			bvhar::HierminnParams minn_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id, grp_mat,
				param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::HierminnInits minn_inits(init_spec);
				sur_objs[i].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
	}
  // Start Gibbs sampling-----------------------------------
	auto run_gibbs = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = sur_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			sur_objs[chain]->doPosteriorDraws(); // alpha -> a -> h -> sigma_h -> h0
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = sur_objs[chain]->returnRecords(num_burn, thin);
		}
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
