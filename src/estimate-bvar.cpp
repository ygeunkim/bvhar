#include "minnesota.h"
#include "bvharinterrupt.h"

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
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd y, int lag,
														int num_chains, int num_iter, int num_burn, int thin,
														Rcpp::List bayes_spec,
														bool include_mean,
														Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	Eigen::MatrixXd response = bvhar::build_y0(y, lag, lag + 1);
	Eigen::MatrixXd design = bvhar::build_x0(y, lag, include_mean);
	bvhar::BvarSpec mn_spec(bayes_spec);
	int dim = response.cols();
	Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		lag, mn_spec._sigma, mn_spec._lambda,
		mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		include_mean
	);
	Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		Eigen::VectorXd::LinSpaced(lag, 1, lag),
		mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	);
	// std::unique_ptr<bvhar::Minnesota> mn_obj(new bvhar::Minnesota(num_iter, design, response, x_dummy, y_dummy, ));
	std::vector<std::unique_ptr<bvhar::Minnesota>> mn_objs(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		mn_objs[i].reset(new bvhar::Minnesota(num_iter, design, response, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
		mn_objs[i]->computePosterior();
	}
	std::vector<Rcpp::List> record(num_chains);
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
			record[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
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
	Rcpp::List res = mn_objs[0]->returnMinnRes();
	res["record"] = Rcpp::wrap(record);
	return res;
	// res.push_back(Rcpp::wrap(record));
	// return res;
	// std::unique_ptr<bvhar::MinnBvar> mn_obj(new bvhar::MinnBvar(y, lag, num_chains, num_iter, mn_spec, include_mean, display_progress, seed_chain));
	// mn_obj->doPosteriorDraws(nthreads, num_burn, thin);
	// return mn_obj->returnMinnRes();
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
Rcpp::List estimate_bvhar_mn(Eigen::MatrixXd y, int week, int month,
														 int num_chains, int num_iter, int num_burn, int thin,
														 Rcpp::List bayes_spec, bool include_mean,
														 Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	Eigen::MatrixXd response = bvhar::build_y0(y, month, month + 1);
	Eigen::MatrixXd var_design = bvhar::build_x0(y, month, include_mean);
	int dim = response.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	Eigen::MatrixXd design = var_design * har_trans.transpose();
	std::vector<std::unique_ptr<bvhar::Minnesota>> mn_objs(num_chains);
	if (bayes_spec.containsElementNamed("delta")) {
		bvhar::BvarSpec mn_spec(bayes_spec);
		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
			3, mn_spec._sigma, mn_spec._lambda,
			mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
			include_mean
		);
		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
			Eigen::VectorXd::LinSpaced(3, 1, 3),
			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		);
		for (int i = 0; i < num_chains; ++i) {
			mn_objs[i].reset(new bvhar::Minnesota(num_iter, design, response, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
			mn_objs[i]->computePosterior();
		}
	} else {
		bvhar::BvharSpec mn_spec(bayes_spec);
		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
			Eigen::VectorXd::LinSpaced(3, 1, 3),
			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		);
		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
			3, mn_spec._sigma, mn_spec._lambda,
			mn_spec._daily, mn_spec._weekly, mn_spec._monthly,
			include_mean
		);
		for (int i = 0; i < num_chains; ++i) {
			mn_objs[i].reset(new bvhar::Minnesota(num_iter, design, response, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
			mn_objs[i]->computePosterior();
		}
	}
	std::vector<Rcpp::List> record(num_chains);
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
			record[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
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
	// Rcpp::List res = mn_objs[0]->returnMinnRes();
	// res.push_back(Rcpp::wrap(record));
	// return res;
	Rcpp::List res = mn_objs[0]->returnMinnRes();
	res["design"] = var_design;
	res["HARtrans"] = har_trans;
	res["record"] = Rcpp::wrap(record);
	return res;
	// std::unique_ptr<bvhar::MinnBvhar> mn_obj;
	// if (minn_short) {
	// 	bvhar::BvarSpec bvhar_spec(bayes_spec);
	// 	// mn_obj = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(y, week, month, num_iter, bvhar_spec, include_mean));
	// 	mn_obj.reset(new bvhar::MinnBvharS(y, week, month, num_chains, num_iter, bvhar_spec, include_mean, display_progress, seed_chain));
	// } else {
	// 	bvhar::BvharSpec bvhar_spec(bayes_spec);
	// 	// mn_obj = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(y, week, month, num_iter, bvhar_spec, include_mean));
	// 	mn_obj.reset(new bvhar::MinnBvharL(y, week, month, num_chains, num_iter, bvhar_spec, include_mean, display_progress, seed_chain));
	// }
	// mn_obj->doPosteriorDraws(nthreads, num_burn, thin);
	// return mn_obj->returnMinnRes();
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
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y,
														int num_chains, int num_iter, int num_burn, int thin,
														Eigen::MatrixXd U,
														Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
  int num_design = y.rows();
  int dim = y.cols();
  if (U.rows() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  if (U.cols() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  // Eigen::MatrixXd prec_mat = (x.transpose() * x + U); // MN precision
  // Eigen::MatrixXd mn_scale_mat = prec_mat.inverse(); // MN scale 1 = inverse of precision
  // Eigen::MatrixXd coef_mat = mn_scale_mat * x.transpose() * y; // MN mean
  // Eigen::MatrixXd yhat = x * coef_mat; // x %*% bhat
  // Eigen::MatrixXd Is = Eigen::MatrixXd::Identity(num_design, num_design);
  // Eigen::MatrixXd scale_mat = y.transpose() * (Is - x * mn_scale_mat * x.transpose()) * y; // IW scale
  // return Rcpp::List::create(
  //   Rcpp::Named("mnmean") = coef_mat,
  //   Rcpp::Named("mnprec") = prec_mat,
  //   Rcpp::Named("fitted") = yhat,
  //   Rcpp::Named("iwscale") = scale_mat,
  //   Rcpp::Named("iwshape") = num_design - dim - 1
  // );
	std::vector<Rcpp::List> record(num_chains);
	std::vector<std::unique_ptr<bvhar::MinnFlat>> flat_objs(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		flat_objs[i].reset(new bvhar::MinnFlat(num_iter, x, y, U, static_cast<unsigned int>(seed_chain[i])));
		flat_objs[i]->computePosterior();
	}
	auto run_mcmc = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					record[chain] = flat_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			flat_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			record[chain] = flat_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_mcmc(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_mcmc(chain);
		}
	}
	Rcpp::List res = flat_objs[0]->returnMinnRes();
	// res.push_back(Rcpp::wrap(record));
	// return res;
	res["record"] = Rcpp::wrap(record);
	return res;
}
