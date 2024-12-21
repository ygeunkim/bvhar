#include <bvharomp.h>
#include <svspillover.h>

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
Rcpp::List dynamic_bvarsv_spillover(int lag, int step, int num_design, Rcpp::List fit_record, bool sparse, bool include_mean, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicSvSpillover>(lag, step, num_design, fit_record, include_mean, sparse, nthreads);
	return spillover->returnSpillover();
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
Rcpp::List dynamic_bvharsv_spillover(int week, int month, int step, int num_design, Rcpp::List fit_record, bool sparse, bool include_mean, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicSvSpillover>(week, month, step, num_design, fit_record, include_mean, sparse, nthreads);
	return spillover->returnSpillover();
}
