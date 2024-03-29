#include "bvharomp.h"
// #include "bvharsim.h"
// #include "bvharstructural.h"
#include "minnspillover.h"

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
	Eigen::VectorXd res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, lag, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		res[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
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
	// Rcpp::List record = sim_mniw(
	// 	num_iter,
	// 	mod["coefficients"],
	// 	prior_prec.inverse(),
	// 	mod["iw_scale"],
	// 	mod["iw_shape"]
	// );
	// Eigen::MatrixXd coef_draw = record["mn"]; // dim_design x num_iter*dim
	// Eigen::MatrixXd coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
	// Eigen::MatrixXd cov_draw = record["iw"]; // dim x num_iter*dim
	// Eigen::MatrixXd cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));
	std::vector<std::vector<Eigen::MatrixXd>> record_warm(num_burn, std::vector<Eigen::MatrixXd>(2));
	std::vector<std::vector<Eigen::MatrixXd>> record(num_iter - num_burn, std::vector<Eigen::MatrixXd>(2));
	for (int i = 0; i < num_burn; i++) {
		record_warm[i] = bvhar::sim_mn_iw(
			mod["coefficients"],
			prior_prec.inverse(),
			mod["iw_scale"],
			mod["iw_shape"]
		);
	}
	for (int i = 0; i < num_iter - num_burn; i++) {
		record[i] = bvhar::sim_mn_iw(
			mod["coefficients"],
			prior_prec.inverse(),
			mod["iw_scale"],
			mod["iw_shape"]
		);
	}
	Eigen::MatrixXd vma_mat(dim * step, dim);
	Eigen::MatrixXd fevd = Eigen::MatrixXd::Zero(dim * step, dim);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads) private(vma_mat)
#endif
	for (int j = 0; j < (num_iter - num_burn); j++) {
		vma_mat = bvhar::convert_vhar_to_vma(
			// coef_burn.middleCols(j * dim, dim),
			record[j][0],
			har_trans,
			step - 1,
			har[1]
		);
		// fevd += bvhar::compute_vma_fevd(vma_mat, cov_burn.middleCols(j * dim, dim), true);
		fevd += bvhar::compute_vma_fevd(vma_mat, record[j][1], true);
	}
	fevd /= (num_iter - num_burn); // add burn-in
	Eigen::MatrixXd spillover = bvhar::compute_sp_index(fevd); // Normalized spillover
  Eigen::VectorXd res(num_horizon);
	res[0] = bvhar::compute_tot(spillover); // Total spillovers
	for (int i = 1; i < num_horizon; i++) {
		roll_mat = y.middleRows(i, window);
		mod = fit(roll_mat, har, bayes_spec, include_mean);
		har_trans = mod["HARtrans"];
		prior_prec = mod["mn_prec"];
		// record = sim_mniw(
		// 	num_iter,
		// 	mod["coefficients"],
		// 	prior_prec.inverse(),
		// 	mod["iw_scale"],
		// 	mod["iw_shape"]
		// );
		// coef_draw = record["mn"]; // dim_design x num_iter*dim
		// coef_burn = coef_draw.rightCols(dim * (num_iter - num_burn));
		// cov_draw = record["iw"]; // dim x num_iter*dim
		// cov_burn = cov_draw.rightCols(dim * (num_iter - num_burn));

		std::vector<std::vector<Eigen::MatrixXd>> record_warm(num_burn, std::vector<Eigen::MatrixXd>(2));
		std::vector<std::vector<Eigen::MatrixXd>> record(num_iter - num_burn, std::vector<Eigen::MatrixXd>(2));
		for (int i = 0; i < num_burn; i++) {
			record_warm[i] = bvhar::sim_mn_iw(
				mod["coefficients"],
				prior_prec.inverse(),
				mod["iw_scale"],
				mod["iw_shape"]
			);
		}
		for (int i = 0; i < num_iter - num_burn; i++) {
			record[i] = bvhar::sim_mn_iw(
				mod["coefficients"],
				prior_prec.inverse(),
				mod["iw_scale"],
				mod["iw_shape"]
			);
		}
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads) private(vma_mat)
	#endif
		for (int j = 0; j < (num_iter - num_burn); j++) {
			vma_mat = bvhar::convert_vhar_to_vma(
				// coef_draw.middleCols(j * dim, dim),
				record[j][0],
				har_trans,
				step - 1,
				har[1]
			);
			// fevd += bvhar::compute_vma_fevd(vma_mat, cov_draw.middleCols(j * dim, dim), true);
			fevd += bvhar::compute_vma_fevd(vma_mat, record[j][1], true);
		}
		fevd /= (num_iter - num_burn); // add burn-in
		spillover = bvhar::compute_sp_index(fevd);
		res[i] = bvhar::compute_tot(spillover);
	}
	return res;
}
