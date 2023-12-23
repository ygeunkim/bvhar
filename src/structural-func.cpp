#include "bvharomp.h"
#include <RcppEigen.h>
#include "fitvar.h"
#include "randsim.h"

//' h-step ahead Forecast Error Variance Decomposition
//' 
//' [w_(h = 1, ij)^T, w_(h = 2, ij)^T, ...]
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize) {
    int dim = cov_mat.cols();
    // Eigen::MatrixXd vma_mat = VARcoeftoVMA(var_coef, var_lag, step);
    int step = vma_coef.rows() / dim; // h-step
    Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd ma_prod(dim, dim);
    Eigen::MatrixXd numer = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd denom = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * step, dim);
    Eigen::MatrixXd cov_diag = Eigen::MatrixXd::Zero(dim, dim);
    cov_diag.diagonal() = 1 / cov_mat.diagonal().cwiseSqrt().array(); // sigma_jj
    for (int i = 0; i < step; i++) {
        ma_prod = vma_coef.block(i * dim, 0, dim, dim).transpose() * cov_mat; // A * Sigma
        innov_account += ma_prod * vma_coef.block(i * dim, 0, dim, dim); // A * Sigma * A^T
        numer.array() += (ma_prod * cov_diag).array().square(); // sum(A * Sigma)_ij / sigma_jj^2
        denom.diagonal() = 1 / innov_account.diagonal().array(); // sigma_jj^(-1) / sum(A * Sigma * A^T)_jj
        res.block(i * dim, 0, dim, dim) = denom * numer; // sigma_jj^(-1) sum(A * Sigma)_ij / sum(A * Sigma * A^T)_jj
    }
    if (normalize) {
        res.array().colwise() /= res.rowwise().sum().array();
    }
    return res;
}

//' h-step ahead Normalized Spillover
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd) {
    return fevd.bottomRows(fevd.cols()) * 100;
}

//' To-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_to_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).colwise().sum();
}

//' From-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_from_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).rowwise().sum();
}

//' Total Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
double compute_tot_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).sum() / spillover.cols();
}

//' Net Pairwise Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_net_spillover(Eigen::MatrixXd spillover) {
    return (spillover.transpose() - spillover) / spillover.cols();
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
Eigen::VectorXd roll_var_tot_spillover(Eigen::MatrixXd y, int window, int step,
																			 int lag, bool include_mean) {
	Rcpp::Function fit("var_lm");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List var_mod = fit(roll_mat, lag, include_mean);
	// Eigen::MatrixXd vma_mat = VARtoVMA(var_mod, step - 1);
	Eigen::MatrixXd vma_mat = VARcoeftoVMA(var_mod["coefficients"], lag, step - 1);
	Eigen::MatrixXd fevd = compute_fevd(vma_mat, var_mod["covmat"], true); // KPPS FEVD
	Eigen::MatrixXd spillover = compute_spillover(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = compute_tot_spillover(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		var_mod = fit(roll_mat, lag, include_mean);
		// vma_mat = VARtoVMA(var_mod, step - 1);
		vma_mat = VARcoeftoVMA(var_mod["coefficients"], lag, step - 1);
		fevd = compute_fevd(vma_mat, var_mod["covmat"], true);
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
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
Eigen::VectorXd roll_vhar_tot_spillover(Eigen::MatrixXd y, int window, int step,
																			  Eigen::VectorXd har, bool include_mean) {
	Rcpp::Function fit("vhar_lm");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List vhar_mod = fit(roll_mat, har, include_mean);
	Eigen::MatrixXd vma_mat = VHARcoeftoVMA(vhar_mod["coefficients"], vhar_mod["HARtrans"], step - 1, vhar_mod["month"]);
	Eigen::MatrixXd fevd = compute_fevd(vma_mat, vhar_mod["covmat"], true); // KPPS FEVD
	Eigen::MatrixXd spillover = compute_spillover(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = compute_tot_spillover(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		vhar_mod = fit(roll_mat, har, include_mean);
		vma_mat = VHARcoeftoVMA(vhar_mod["coefficients"], vhar_mod["HARtrans"], step - 1, vhar_mod["month"]);
		fevd = compute_fevd(vma_mat, vhar_mod["covmat"], true);
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
	}
	return res;
}

//' Rolling-sample Total Spillover Index of BVAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param lag BVAR order
//' @param bayes_spec BVAR specification
//' @param include_mean Add constant term
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd roll_bvar_tot_spillover(Eigen::MatrixXd y, int window, int step,
																				int num_iter, int num_burn,
																			  int lag, Rcpp::List bayes_spec, bool include_mean, int nthreads) {
	Rcpp::Function fit("bvar_minnesota");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List mod = fit(roll_mat, lag, bayes_spec, include_mean);
	// Eigen::MatrixXd prior_mean = mod["coefficients"];
	Eigen::MatrixXd prior_prec = mod["mn_prec"];
	Rcpp::List record = sim_mniw(
		num_iter,
		mod["coefficients"],
		prior_prec.inverse(),
		mod["iw_scale"],
		mod["iw_shape"]
	);
	Eigen::MatrixXd coef_draw = record["mn"]; // dim_design x num_iter*dim
	Eigen::MatrixXd coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
	Eigen::MatrixXd cov_draw = record["iw"]; // dim x num_iter*dim
	Eigen::MatrixXd cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));
	Eigen::MatrixXd vma_mat(dim * step, dim);
	Eigen::MatrixXd fevd = Eigen::MatrixXd::Zero(dim * step, dim);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) private(vma_mat)
	for (int j = 0; j < (num_iter - num_burn); j++) {
		vma_mat = VARcoeftoVMA(
			coef_burn.middleCols(j * dim, dim),
			lag,
			step - 1
		);
		fevd += compute_fevd(vma_mat, cov_burn.middleCols(j * dim, dim), true);
	}
#else
	for (int j = 0; j < (num_iter - num_burn); j++) {
		vma_mat = VARcoeftoVMA(
			coef_burn.middleCols(j * dim, dim),
			lag,
			step - 1
		);
		fevd += compute_fevd(vma_mat, cov_burn.middleCols(j * dim, dim), true);
	}
#endif
	fevd /= (num_iter - num_burn);
	Eigen::MatrixXd spillover = compute_spillover(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = compute_tot_spillover(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		mod = fit(roll_mat, lag, bayes_spec, include_mean);
		prior_prec = mod["mn_prec"];
		record = sim_mniw(
			num_iter,
			mod["coefficients"],
			prior_prec.inverse(),
			mod["iw_scale"],
			mod["iw_shape"]
		);
		coef_draw = record["mn"]; // dim_design x num_iter*dim
		coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
		cov_draw = record["iw"]; // dim x num_iter*dim
		cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads) private(vma_mat)
		for (int j = 0; j < (num_iter - num_burn); j++) {
			vma_mat = VARcoeftoVMA(
				coef_draw.middleCols(j * dim, dim),
				lag,
				step - 1
			);
			fevd += compute_fevd(vma_mat, cov_draw.middleCols(j * dim, dim), true);
		}
	#else
		for (int j = 0; j < (num_iter - num_burn); j++) {
			vma_mat = VARcoeftoVMA(
				coef_draw.middleCols(j * dim, dim),
				lag,
				step - 1
			);
			fevd += compute_fevd(vma_mat, cov_draw.middleCols(j * dim, dim), true);
		}
	#endif
		fevd /= (num_iter - num_burn);
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
	}
	return res;
}

//' Rolling-sample Total Spillover Index of BVHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param har BVHAR order
//' @param bayes_spec BVHAR specification
//' @param include_mean Add constant term
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd roll_bvhar_tot_spillover(Eigen::MatrixXd y, int window, int step,
																				 int num_iter, int num_burn,
																			   Eigen::VectorXd har, Rcpp::List bayes_spec, bool include_mean, int nthreads) {
	Rcpp::Function fit("bvhar_minnesota");
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd roll_mat = y.topRows(window);
	Rcpp::List mod = fit(roll_mat, har, bayes_spec, include_mean);
	Eigen::MatrixXd har_trans = mod["HARtrans"];
	Eigen::MatrixXd prior_prec = mod["mn_prec"];
	Rcpp::List record = sim_mniw(
		num_iter,
		mod["coefficients"],
		prior_prec.inverse(),
		mod["iw_scale"],
		mod["iw_shape"]
	);
	Eigen::MatrixXd coef_draw = record["mn"]; // dim_design x num_iter*dim
	Eigen::MatrixXd coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
	Eigen::MatrixXd cov_draw = record["iw"]; // dim x num_iter*dim
	Eigen::MatrixXd cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));
	Eigen::MatrixXd vma_mat(dim * step, dim);
	Eigen::MatrixXd fevd = Eigen::MatrixXd::Zero(dim * step, dim);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) private(vma_mat)
	for (int j = 0; j < (num_iter - num_burn); j++) {
		vma_mat = VHARcoeftoVMA(
			coef_burn.middleCols(j * dim, dim),
			har_trans,
			step - 1,
			har[1]
		);
		fevd += compute_fevd(vma_mat, cov_burn.middleCols(j * dim, dim), true);
	}
#else
	for (int j = 0; j < (num_iter - num_burn); j++) {
		vma_mat = VHARcoeftoVMA(
			coef_burn.middleCols(j * dim, dim),
			har_trans,
			step - 1,
			har[1]
		);
		fevd += compute_fevd(vma_mat, cov_burn.middleCols(j * dim, dim), true);
	}
#endif
	fevd /= (num_iter - num_burn); // add burn-in
	Eigen::MatrixXd spillover = compute_spillover(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = compute_tot_spillover(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		mod = fit(roll_mat, har, bayes_spec, include_mean);
		har_trans = mod["HARtrans"];
		prior_prec = mod["mn_prec"];
		record = sim_mniw(
			num_iter,
			mod["coefficients"],
			prior_prec.inverse(),
			mod["iw_scale"],
			mod["iw_shape"]
		);
		coef_draw = record["mn"]; // dim_design x num_iter*dim
		coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
		cov_draw = record["iw"]; // dim x num_iter*dim
		cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads) private(vma_mat)
		for (int j = 0; j < (num_iter - num_burn); j++) {
			vma_mat = VHARcoeftoVMA(
				coef_draw.middleCols(j * dim, dim),
				har_trans,
				step - 1,
				har[1]
			);
			fevd += compute_fevd(vma_mat, cov_draw.middleCols(j * dim, dim), true);
		}
	#else
		for (int j = 0; j < (num_iter - num_burn); j++) {
			vma_mat = VHARcoeftoVMA(
				coef_draw.middleCols(j * dim, dim),
				har_trans,
				step - 1,
				har[1]
			);
			fevd += compute_fevd(vma_mat, cov_draw.middleCols(j * dim, dim), true);
		}
	#endif
		fevd /= (num_iter - num_burn); // add burn-in
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
	}
	return res;
}