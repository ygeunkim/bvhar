#include "bvharomp.h"
#include <RcppEigen.h>
#include "structural.h"
#include "bvhardraw.h"

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
Eigen::VectorXd dynamic_vhar_tot_spillover(Eigen::MatrixXd y, int window, int step, Eigen::VectorXd har, bool include_mean) {
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
Eigen::VectorXd dynamic_bvar_tot_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn,
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
Eigen::VectorXd dynamic_bvhar_tot_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn,
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

//' Dynamic Total Spillover Index of BVAR-SV
//' 
//' @param lag VAR lag.
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param response_mat Response matrix.
//' @param phi_record Coefficients MCMC record
//' @param h_record log volatility MCMC record
//' @param a_record Contemporaneous coefficients MCMC record
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_bvarsv_tot_spillover(int lag, int step, Eigen::MatrixXd response_mat,
																						 Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, int nthreads) {
	int num_sim = alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
	int dim_design = alpha_record.cols() / dim;
	Eigen::MatrixXd vma_mat(dim * step, dim);
	Eigen::MatrixXd fevd = Eigen::MatrixXd::Zero(dim * step, dim);
	Eigen::MatrixXd spillover(dim, dim);
	Eigen::VectorXd res(num_design); // length = T - p
	Eigen::MatrixXd h_time(num_sim, dim); // h_t = (h_t1, ..., h_tk)
	// Eigen::MatrixXd contem_inv = Eigen::MatrixXd::Zero(dim, dim); // L^(-1)
  Eigen::MatrixXd lvol_sqrt = Eigen::MatrixXd::Zero(dim, dim); // D_t with h_t = (h_t1, ..., h_tk)
  Eigen::MatrixXd tvp_sig(dim, dim); // Sigma_t
	Eigen::MatrixXd sqrt_sig(dim, dim);
	for (int i = 0; i < num_design; i++) {
		fevd = Eigen::MatrixXd::Zero(dim * step, dim);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads) private(vma_mat, tvp_sig, sqrt_sig, lvol_sqrt)
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal();
			sqrt_sig = build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = VARcoeftoVMA(unvectorize(alpha_record.row(j), dim_design, dim), lag, step - 1);
			#pragma omp critical
			fevd += compute_fevd(vma_mat, tvp_sig, true);
		}
	#else
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal(); // D_t^(1 / 2) = diag(exp(h_t / 2))
			sqrt_sig = build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = VARcoeftoVMA(unvectorize(alpha_record.row(j), dim_design, dim), lag, step - 1);
			fevd += compute_fevd(vma_mat, tvp_sig, true);
		}
	#endif
		fevd /= num_sim;
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
	}
	return res;
}

//' Dynamic Total Spillover Index of BVHAR-SV
//' 
//' @param month VHAR month order.
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param response_mat Response matrix.
//' @param HARtrans VHAR linear transformation matrix
//' @param phi_record Coefficients MCMC record
//' @param h_record log volatility MCMC record
//' @param a_record Contemporaneous coefficients MCMC record
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_bvharsv_tot_spillover(int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
																							Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, int nthreads) {
	int num_sim = phi_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_har = HARtrans.rows();
	Eigen::MatrixXd vma_mat(dim * step, dim);
	Eigen::MatrixXd fevd = Eigen::MatrixXd::Zero(dim * step, dim);
	Eigen::MatrixXd spillover(dim, dim);
	Eigen::VectorXd res(num_design); // length = T - month
	Eigen::MatrixXd h_time(num_sim, dim); // h_t = (h_t1, ..., h_tk)
  Eigen::MatrixXd lvol_sqrt = Eigen::MatrixXd::Zero(dim, dim); // D_t^(1 / 2)
  Eigen::MatrixXd tvp_sig(dim, dim); // Sigma_t
	Eigen::MatrixXd sqrt_sig(dim, dim);
	for (int i = 0; i < num_design; i++) {
		h_time = h_record.middleCols(i * dim, dim);
		fevd = Eigen::MatrixXd::Zero(dim * step, dim);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads) private(vma_mat, tvp_sig, sqrt_sig, lvol_sqrt)
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal();
			sqrt_sig = build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = VHARcoeftoVMA(unvectorize(phi_record.row(j), dim_har, dim), HARtrans, step - 1, month);
			#pragma omp critical
			fevd += compute_fevd(vma_mat, tvp_sig, true);
		}
	#else
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal();
			sqrt_sig = build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = VHARcoeftoVMA(unvectorize(phi_record.row(j), dim_har, dim), HARtrans, step - 1, month);
			fevd += compute_fevd(vma_mat, tvp_sig, true);
		}
	#endif
		fevd /= num_sim;
		spillover = compute_spillover(fevd);
		res[i] = compute_tot_spillover(spillover);
	}
	return res;
}
