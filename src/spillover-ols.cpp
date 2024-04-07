#include "bvharomp.h"
#include "olsspillover.h"

//' Generalized Spillover of VAR
//' 
//' @param object varlse or vharlse object.
//' @param step Step to forecast.
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_ols_spillover(Rcpp::List object, int step) {
	if (!(object.inherits("varlse") || object.inherits("vharlse"))) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
	std::unique_ptr<Eigen::MatrixXd> coef_mat;
	int ord;
	if (object.inherits("vharlse")) {
		coef_mat.reset(new Eigen::MatrixXd(Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]).transpose() * Rcpp::as<Eigen::MatrixXd>(object["coefficients"])));
		ord = object["month"];
	} else {
		coef_mat.reset(new Eigen::MatrixXd(Rcpp::as<Eigen::MatrixXd>(object["coefficients"])));
		ord = object["p"];
	}
	bvhar::StructuralFit fit(*coef_mat, ord, step - 1, Rcpp::as<Eigen::MatrixXd>(object["covmat"]));
	std::unique_ptr<bvhar::OlsSpillover> spillover(new bvhar::OlsSpillover(fit));
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

//' Rolling-sample Total Spillover Index of VAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param lag VAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_var_spillover(Eigen::MatrixXd y, int window, int step, int lag, bool include_mean, int method, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVar>(new bvhar::OlsVar(roll_mat, lag, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

//' Rolling-sample Total Spillover Index of VHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param har VHAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_vhar_spillover(Eigen::MatrixXd y, int window, int step, int week, int month, bool include_mean, int method, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVhar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVhar>(new bvhar::OlsVhar(roll_mat, week, month, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

