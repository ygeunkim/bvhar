#include "bvharomp.h"
#include "bvhardraw.h"
#include "bvharstructural.h"

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
	// int dim_design = alpha_record.cols() / dim;
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
			sqrt_sig = bvhar::build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = bvhar::convert_var_to_vma(bvhar::unvectorize(alpha_record.row(j), dim), lag, step - 1);
			#pragma omp critical
			fevd += bvhar::compute_vma_fevd(vma_mat, tvp_sig, true);
		}
	#else
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal(); // D_t^(1 / 2) = diag(exp(h_t / 2))
			sqrt_sig = bvhar::build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = bvhar::convert_var_to_vma(bvhar::unvectorize(alpha_record.row(j), dim), lag, step - 1);
			fevd += bvhar::compute_vma_fevd(vma_mat, tvp_sig, true);
		}
	#endif
		fevd /= num_sim;
		spillover = bvhar::compute_sp_index(fevd);
		res[i] = bvhar::compute_tot(spillover);
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
  // int dim_har = HARtrans.rows();
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
			sqrt_sig = bvhar::build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = bvhar::convert_vhar_to_vma(bvhar::unvectorize(phi_record.row(j), dim), HARtrans, step - 1, month);
			#pragma omp critical
			fevd += bvhar::compute_vma_fevd(vma_mat, tvp_sig, true);
		}
	#else
		for (int j = 0; j < num_sim; j++) {
			lvol_sqrt = (h_time.row(j) / 2).array().exp().matrix().asDiagonal();
			sqrt_sig = bvhar::build_inv_lower(dim, a_record.row(j)).triangularView<Eigen::Lower>().solve(lvol_sqrt);
			tvp_sig = sqrt_sig * sqrt_sig.transpose(); // Sigma_t = L^(-1) D_t (L^T)^(-1)
			vma_mat = bvhar::convert_vhar_to_vma(bvhar::unvectorize(phi_record.row(j), dim), HARtrans, step - 1, month);
			fevd += bvhar::compute_vma_fevd(vma_mat, tvp_sig, true);
		}
	#endif
		fevd /= num_sim;
		spillover = bvhar::compute_sp_index(fevd);
		res[i] = bvhar::compute_tot(spillover);
	}
	return res;
}
