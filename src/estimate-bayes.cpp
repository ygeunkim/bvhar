#include <bvhar/bayesfit>

//' BVAR(p) Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param lag VAR order
//' @param bayes_spec BVAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean) {
	bvhar::BvarSpec mn_spec(bayes_spec);
	std::unique_ptr<bvhar::MinnBvar> mn_obj(new bvhar::MinnBvar(y, lag, mn_spec, include_mean));
	return mn_obj->returnMinnRes();
}

//' BVHAR Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param week VHAR week order
//' @param month VHAR month order
//' @param bayes_spec BVHAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvhar_mn(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean) {
	std::unique_ptr<bvhar::MinnBvhar> mn_obj;
	if (bayes_spec.containsElementNamed("delta")) {
		bvhar::BvarSpec bvhar_spec(bayes_spec);
		mn_obj.reset(new bvhar::MinnBvharS(y, week, month, bvhar_spec, include_mean));
	} else {
		bvhar::BvharSpec bvhar_spec(bayes_spec);
		mn_obj.reset(new bvhar::MinnBvharL(y, week, month, bvhar_spec, include_mean));
	}
	return mn_obj->returnMinnRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mh(int num_chains, int num_iter, int num_burn, int thin,
														Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy,
														Rcpp::List param_prior, Rcpp::List param_init,
														Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	std::vector<std::unique_ptr<bvhar::MhMinnesota>> mn_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	Rcpp::List lambda_spec = param_prior["lambda"];
	Rcpp::List psi_spec = param_prior["sigma"];
	bvhar::MhMinnSpec mn_spec(lambda_spec, psi_spec);
  for (int i = 0; i < num_chains; ++i) {
		Rcpp::List init_spec = param_init[i];
		bvhar::MhMinnInits mn_init(init_spec);
		mn_objs[i].reset(new bvhar::MhMinnesota(num_iter, mn_spec, mn_init, x, y, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
		mn_objs[i]->computePosterior();
	}
	auto run_mh = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mn_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			mn_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_mh(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_mh(chain);
		}
	}
	return Rcpp::wrap(res);
}

//' BVAR(p) Point Estimates based on Nonhierarchical Matrix Normal Prior
//' 
//' Point estimates for Ghosh et al. (2018) nonhierarchical model for BVAR.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param U Positive definite matrix, covariance matrix corresponding to the column of the model parameter B
//' 
//' @details
//' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  if (U.rows() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  if (U.cols() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
	std::unique_ptr<bvhar::MinnFlat> mn_obj(new bvhar::MinnFlat(x, y, U));
	return mn_obj->returnMinnRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mniw(int num_chains, int num_iter, int num_burn, int thin,
												 const Eigen::MatrixXd& mn_mean, const Eigen::MatrixXd& mn_prec,
												 const Eigen::MatrixXd& iw_scale, double iw_shape,
												 Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	std::vector<std::unique_ptr<bvhar::McmcMniw>> mn_objs(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		bvhar::MinnFit mn_fit(mn_mean, mn_prec, iw_scale, iw_shape);
		mn_objs[i].reset(new bvhar::McmcMniw(num_iter, mn_fit, static_cast<unsigned int>(seed_chain[i])));
	}
	std::vector<Rcpp::List> res(num_chains);
	auto run_conj = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		for (int i = 0; i < num_iter; ++i) {
			bar.increment();
			bar.update();
			mn_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_conj(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			run_conj(chain);
		}
	}
	return Rcpp::wrap(res);
}

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
												Rcpp::List param_init, int prior_type, bool ggl,
                        Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
                        bool include_mean, Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = [&]() -> std::unique_ptr<bvhar::McmcInterface> {
		if (param_reg.containsElementNamed("initial_mean")) {
			if (ggl) {
				return std::make_unique<bvhar::McmcRun<bvhar::McmcSv, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads
				);
			}
			return std::make_unique<bvhar::McmcRun<bvhar::McmcSv, false>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		if (ggl) {
			return std::make_unique<bvhar::McmcRun<bvhar::McmcReg, true>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		return std::make_unique<bvhar::McmcRun<bvhar::McmcReg, false>>(
			num_chains, num_iter, num_burn, thin, x, y,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain,
			display_progress, nthreads
		);
	}();
  // Start Gibbs sampling-----------------------------------
	return mcmc_run->returnRecords();
}
