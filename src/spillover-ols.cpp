#include "bvharomp.h"
#include "bvharstructural.h"

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
Eigen::VectorXd dynamic_var_tot_spillover(Eigen::MatrixXd y, int window, int step, int lag, bool include_mean) {
	Rcpp::Function fit("var_lm");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List var_mod = fit(roll_mat, lag, include_mean);
	// Eigen::MatrixXd vma_mat = VARtoVMA(var_mod, step - 1);
	Eigen::MatrixXd vma_mat = bvhar::convert_var_to_vma(var_mod["coefficients"], lag, step - 1);
	Eigen::MatrixXd fevd = bvhar::compute_vma_fevd(vma_mat, var_mod["covmat"], true); // KPPS FEVD
	Eigen::MatrixXd spillover = bvhar::compute_sp_index(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = bvhar::compute_tot(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		var_mod = fit(roll_mat, lag, include_mean);
		// vma_mat = VARtoVMA(var_mod, step - 1);
		vma_mat = bvhar::convert_var_to_vma(var_mod["coefficients"], lag, step - 1);
		fevd = bvhar::compute_vma_fevd(vma_mat, var_mod["covmat"], true);
		spillover = bvhar::compute_sp_index(fevd);
		res[i] = bvhar::compute_tot(spillover);
	}
	return res;
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
Eigen::VectorXd dynamic_vhar_tot_spillover(Eigen::MatrixXd y, int window, int step, Eigen::VectorXd har, bool include_mean) {
	Rcpp::Function fit("vhar_lm");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List vhar_mod = fit(roll_mat, har, include_mean);
	Eigen::MatrixXd vma_mat = bvhar::convert_vhar_to_vma(vhar_mod["coefficients"], vhar_mod["HARtrans"], step - 1, vhar_mod["month"]);
	Eigen::MatrixXd fevd = bvhar::compute_vma_fevd(vma_mat, vhar_mod["covmat"], true); // KPPS FEVD
	Eigen::MatrixXd spillover = bvhar::compute_sp_index(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = bvhar::compute_tot(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		vhar_mod = fit(roll_mat, har, include_mean);
		vma_mat = bvhar::convert_vhar_to_vma(vhar_mod["coefficients"], vhar_mod["HARtrans"], step - 1, vhar_mod["month"]);
		fevd = bvhar::compute_vma_fevd(vma_mat, vhar_mod["covmat"], true);
		spillover = bvhar::compute_sp_index(fevd);
		res[i] = bvhar::compute_tot(spillover);
	}
	return res;
}

