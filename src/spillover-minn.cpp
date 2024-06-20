// #include "bvharomp.h"
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
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["covmat"]), object["iw_shape"]);
		spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
	} else {
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["covmat"]), object["iw_shape"]);
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

// Rcpp::List compute_bvarmn_spillover(int lag, int step, Eigen::MatrixXd alpha_record, Eigen::MatrixXd sig_record) {
// 	// if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
//   //   Rcpp::stop("'object' must be bvarmn or bvharmn object.");
//   // }
// 	bvhar::MinnRecords mn_record(alpha_record, sig_record);
// 	std::unique_ptr<bvhar::MinnSpillover> spillover(new bvhar::MinnSpillover(mn_record, step, lag));
// 	// std::unique_ptr<bvhar::MinnSpillover> spillover;
// 	// if (object.inherits("bvharmn")) {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
// 	// } else {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
// 	// }
// 	// spillover->updateMniw();
// 	spillover->computeSpillover();
// 	Eigen::VectorXd to_sp = spillover->returnTo();
// 	Eigen::VectorXd from_sp = spillover->returnFrom();
// 	return Rcpp::List::create(
// 		Rcpp::Named("connect") = spillover->returnSpillover(),
// 		Rcpp::Named("to") = to_sp,
// 		Rcpp::Named("from") = from_sp,
// 		Rcpp::Named("tot") = spillover->returnTot(),
// 		Rcpp::Named("net") = to_sp - from_sp,
// 		Rcpp::Named("net_pairwise") = spillover->returnNet()
// 	);
// }

// Rcpp::List compute_bvharmn_spillover(int month, int step, Eigen::MatrixXd har_trans, Eigen::MatrixXd phi_record, Eigen::MatrixXd sig_record) {
// 	// if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
//   //   Rcpp::stop("'object' must be bvarmn or bvharmn object.");
//   // }
// 	bvhar::MinnRecords mn_record(phi_record, sig_record);
// 	std::unique_ptr<bvhar::BvharSpillover> spillover(new bvhar::BvharSpillover(mn_record, step, month, har_trans));
// 	// std::unique_ptr<bvhar::MinnSpillover> spillover;
// 	// if (object.inherits("bvharmn")) {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
// 	// } else {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
// 	// }
// 	// spillover->updateMniw();
// 	spillover->computeSpillover();
// 	Eigen::VectorXd to_sp = spillover->returnTo();
// 	Eigen::VectorXd from_sp = spillover->returnFrom();
// 	return Rcpp::List::create(
// 		Rcpp::Named("connect") = spillover->returnSpillover(),
// 		Rcpp::Named("to") = to_sp,
// 		Rcpp::Named("from") = from_sp,
// 		Rcpp::Named("tot") = spillover->returnTot(),
// 		Rcpp::Named("net") = to_sp - from_sp,
// 		Rcpp::Named("net_pairwise") = spillover->returnNet()
// 	);
// }

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
// Rcpp::List dynamic_bvar_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin,
// 																 	int lag, Rcpp::List bayes_spec, bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	int dim = y.cols();
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		// Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, lag, lag + 1);
		// Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, lag, include_mean);
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i] = std::unique_ptr<bvhar::MinnBvar>(new bvhar::MinnBvar(roll_mat, lag, mn_spec, include_mean));
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	lag, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		// 	include_mean
		// );
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(lag, 1, lag),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
		// for (int j = 0; j < num_chains; ++j) {
		// 	mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
		// 	mn_objs[i][j]->computePosterior();
		// }
	}
	// std::vector<std::vector<bvhar::MinnRecords>> mn_recs(num_horizon);
	// for (auto &rec_chain : mn_recs) {
	// 	rec_chain.resize(num_chains);
	// 	for (auto &ptr : rec_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<bvhar::MinnRecords> mn_rec(num_chains);
	std::vector<std::unique_ptr<bvhar::MinnSpillover>> spillover(num_horizon);
	// std::vector<bvhar::MinnRecords> mn_recs(num_chains);
	// std::vector<Eigen::MatrixXd> coef_record(num_chains);
	// std::vector<Eigen::MatrixXd> sig_record(num_chains);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, dim);
	Eigen::MatrixXd from_sp(num_horizon, dim);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
	// for (int win = 0; win < num_horizon; ++win) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, thin, lag, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr

		// std::vector<Eigen::MatrixXd> coef_record(num_chains);
		// std::vector<Eigen::MatrixXd> sig_record(num_chains);
		// for (int chain = 0; chain < num_chains; ++chain) {
		// 	for (int i = 0; i < num_iter; ++i) {
		// 		mn_objs[win][chain]->doPosteriorDraws();
		// 	}
		// 	// mn_recs[i][j] = mn_objs[i][j]->returnMinnRecords(num_burn, thin);
		// 	bvhar::MinnRecords mn_rec = mn_objs[win][chain]->returnMinnRecords(num_burn, thin);
		// 	// mn_recs[j] = mn_objs[i][j]->returnMinnRecords(num_burn, thin);
		// 	coef_record[chain] = mn_rec.coef_record;
		// 	sig_record[chain] = mn_rec.sig_record;
		// 	mn_objs[win][chain].reset();
		// }
		// Eigen::MatrixXd coef = std::accumulate(
		// 	coef_record.begin() + 1, coef_record.end(), coef_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// Eigen::MatrixXd sig = std::accumulate(
		// 	sig_record.begin() + 1, sig_record.end(), sig_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// bvhar::MinnRecords mn_record(coef, sig);
		// spillover[win].reset(new bvhar::MinnSpillover(mn_record, step, lag));
		// // bvhar::MinnFit mn_fit = mn_objs[win]->returnMinnFit();
		// // spillover[win].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, thin, lag, static_cast<unsigned int>(seed_chain[win])));
		// // spillover[win]->updateMniw();
		// spillover[win]->computeSpillover();
		// to_sp.row(win) = spillover[win]->returnTo();
		// from_sp.row(win) = spillover[win]->returnFrom();
		// tot[win] = spillover[win]->returnTot();
		// // mn_objs[win].reset(); // free the memory by making nullptr
		// spillover[win].reset(); // free the memory by making nullptr
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
// Rcpp::List dynamic_bvhar_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin,
// 																	 int week, int month, Rcpp::List bayes_spec, bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// for (int i = 0; i < num_horizon; ++i) {
	// 	Eigen::MatrixXd roll_mat = y.middleRows(i, window);
	// 	Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, month, month + 1);
	// 	Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, month, include_mean) * har_trans.transpose();
		// bvhar::BvarSpec mn_spec(bayes_spec);
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	lag, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		// 	include_mean
		// );
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(lag, 1, lag),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
	// 	if (bayes_spec.containsElementNamed("delta")) {
	// 		// bvhar::BvarSpec bvhar_spec(bayes_spec);
	// 		bvhar::BvarSpec mn_spec(bayes_spec);
	// 		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 			3, mn_spec._sigma, mn_spec._lambda,
	// 			mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 			include_mean
	// 		);
	// 		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 			Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 		);
	// 		// mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat, week, month, bvhar_spec, include_mean));
	// 		for (int j = 0; j < num_chains; ++j) {
	// 			mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
	// 			mn_objs[i][j]->computePosterior();
	// 		}
	// 	} else {
	// 		// bvhar::BvharSpec bvhar_spec(bayes_spec);
	// 		bvhar::BvharSpec mn_spec(bayes_spec);
	// 		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 			3, mn_spec._sigma, mn_spec._lambda,
	// 			mn_spec._daily, mn_spec._weekly, mn_spec._monthly,
	// 			include_mean
	// 		);
	// 		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 			Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 		);
	// 		// mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat, week, month, bvhar_spec, include_mean));
	// 		for (int j = 0; j < num_chains; ++j) {
	// 			mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
	// 			mn_objs[i][j]->computePosterior();
	// 		}
	// 	}
	// }
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
	// std::vector<bvhar::MinnRecords> mn_recs(num_chains);
	// std::vector<Eigen::MatrixXd> coef_record(num_chains);
	// std::vector<Eigen::MatrixXd> sig_record(num_chains);
  Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
	// for (int win = 0; win < num_horizon; ++win) {
		// std::vector<Eigen::MatrixXd> coef_record(num_chains);
		// std::vector<Eigen::MatrixXd> sig_record(num_chains);
		// for (int chain = 0; chain < num_chains; ++chain) {
		// 	for (int i = 0; i < num_iter; ++i) {
		// 		mn_objs[win][chain]->doPosteriorDraws();
		// 	}
		// 	bvhar::MinnRecords mn_rec = mn_objs[win][chain]->returnMinnRecords(num_burn, thin);
		// 	coef_record[chain] = mn_rec.coef_record;
		// 	sig_record[chain] = mn_rec.sig_record;
		// 	mn_objs[win][chain].reset();
		// }
		// Eigen::MatrixXd coef = std::accumulate(
		// 	coef_record.begin() + 1, coef_record.end(), coef_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// Eigen::MatrixXd sig = std::accumulate(
		// 	sig_record.begin() + 1, sig_record.end(), sig_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// bvhar::MinnRecords mn_record(coef, sig);
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		// spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, thin, month, har_trans, static_cast<unsigned int>(seed_chain[win])));
		spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, thin, month, har_trans, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		// spillover[win].reset(new bvhar::BvharSpillover(mn_record, step, month, har_trans));
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
