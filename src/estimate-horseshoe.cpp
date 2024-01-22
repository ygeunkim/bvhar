#include <mcmchs.h>
#include <bvharprogress.h>
#include <bvharinterrupt.h>

//' Gibbs Sampler for Horseshoe BVAR SUR Parameterization
//' 
//' This function conducts Gibbs sampling for horseshoe prior BVAR(p).
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_priorvar Initial variance constant
//' @param init_local Initial local shrinkage hyperparameters
//' @param init_global Initial global shrinkage hyperparameter
//' @param init_sigma Initial sigma
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param fast Fast sampling?
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_sur_horseshoe(int num_chains, int num_iter, int num_burn, int thin,
                                  Eigen::MatrixXd x, Eigen::MatrixXd y,
                                  Eigen::VectorXd init_local,
                                  Eigen::VectorXd init_global,
                                  double init_sigma,
                                  Eigen::VectorXi grp_id,
                                  Eigen::MatrixXi grp_mat,
                                  int blocked_gibbs,
                                  bool fast,
																	Eigen::VectorXi seed_chain,
                                  bool display_progress, int nthreads) {
  #ifdef _OPENMP
		Eigen::setNbThreads(nthreads);
	#endif
	std::vector<std::unique_ptr<McmcHs>> hs_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	HsParams hs_params(
		num_iter, x, y, init_local, init_global, init_sigma,
		grp_id, grp_mat
	);
	switch (blocked_gibbs) {
	case 1: {
		if (fast) {
			for (int i = 0; i < num_chains; i++) {
				hs_objs[i] = std::unique_ptr<McmcHs>(new FastHs(hs_params, static_cast<unsigned int>(seed_chain[i])));
			}
		} else {
			for (int i = 0; i < num_chains; i++) {
				hs_objs[i] = std::unique_ptr<McmcHs>(new McmcHs(hs_params, static_cast<unsigned int>(seed_chain[i])));
			}
		}
		break;
	}
	case 2:
		for (int i = 0; i < num_chains; i++) {
			hs_objs[i] = std::unique_ptr<McmcHs>(new BlockHs(hs_params, static_cast<unsigned int>(seed_chain[i])));
		}
	}
  // Start Gibbs sampling-----------------------------------
	auto run_gibbs = [&](int chain) {
		bvharprogress bar(num_iter, display_progress);
		bvharinterrupt();
		for (int i = 1; i < num_iter + 1; i++) {
			if (bvharinterrupt::is_interrupted()) {
				res[chain] = hs_objs[chain]->returnRecords(0, 1);
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			hs_objs[chain]->addStep();
			hs_objs[chain]->doPosteriorDraws(); // alpha -> sigma -> nuj -> xi -> lambdaj -> tau
		}
		res[chain] = hs_objs[chain]->returnRecords(num_burn, thin);
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
