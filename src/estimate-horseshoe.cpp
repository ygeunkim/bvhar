#include "mcmchs.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

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
Rcpp::List estimate_sur_horseshoe(int num_iter, int num_burn,
                                  Eigen::MatrixXd x, Eigen::MatrixXd y,
                                  Eigen::VectorXd init_local,
                                  Eigen::VectorXd init_global,
                                  double init_sigma,
                                  Eigen::VectorXi grp_id,
                                  Eigen::MatrixXd grp_mat,
                                  int blocked_gibbs,
                                  bool fast,
                                  bool display_progress) {
  HsParams hs_params(
		num_iter, x, y, init_local, init_global, init_sigma,
		grp_id, grp_mat
	);
	std::unique_ptr<McmcHs> hs_obj;
	switch (blocked_gibbs) {
	case 1: {
		if (fast) {
			hs_obj = std::unique_ptr<McmcHs>(new FastHs(hs_params));
		} else {
			hs_obj = std::unique_ptr<McmcHs>(new McmcHs(hs_params));
		}
		break;
	}
	case 2:
		hs_obj = std::unique_ptr<McmcHs>(new BlockHs(hs_params));
	}
  // Start Gibbs sampling-----------------------------------
  bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
  for (int i = 1; i < num_iter + 1; i++) {
    if (bvharinterrupt::is_interrupted()) {
			return hs_obj->returnRecords(0);
    }
    bar.increment();
		if (display_progress) {
			bar.update();
		}
		hs_obj->addStep();
		hs_obj->doPosteriorDraws(); // alpha -> sigma -> nuj -> xi -> lambdaj -> tau
  }
	return hs_obj->returnRecords(num_burn);
}
