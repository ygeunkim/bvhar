#include "bvharomp.h"
#include "minnspillover.h"

//' Generalized Spillover of Minnesota prior
//' 
//' @param object varlse or vharlse object.
//' @param step Step to forecast.
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param seed Random seed for boost library
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_mn_spillover(Rcpp::List object, int step, int num_iter, int num_burn, int thin, unsigned int seed) {
	if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
    Rcpp::stop("'object' must be bvarmn or bvharmn object.");
  }
	std::unique_ptr<bvhar::MinnSpillover> spillover;
	if (object.inherits("bvharmn")) {
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
		spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
	} else {
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
		spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
	}
	spillover->updateMniw();
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}

//' Rolling-sample Total Spillover Index of BVAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param lag BVAR order
//' @param bayes_spec BVAR specification
//' @param include_mean Add constant term
//' @param seed_chain Random seed for each window
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvar_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																 	int lag, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i] = std::unique_ptr<bvhar::MinnBvar>(new bvhar::MinnBvar(roll_mat, lag, mn_spec, include_mean));
	}
	std::vector<std::unique_ptr<bvhar::MinnSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, thin, lag, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

//' Rolling-sample Total Spillover Index of BVHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param week Week order
//' @param month Month order
//' @param bayes_spec BVHAR specification
//' @param include_mean Add constant term
//' @param seed_chain Random seed for each window
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvhar_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																			 int week, int month, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		if (bayes_spec.containsElementNamed("delta")) {
			bvhar::BvarSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat, week, month, bvhar_spec, include_mean));
		} else {
			bvhar::BvharSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat, week, month, bvhar_spec, include_mean));
		}
	}
	std::vector<std::unique_ptr<bvhar::BvharSpillover>> spillover(num_horizon);
  Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, thin, month, har_trans, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}
