// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// build_y0
Eigen::MatrixXd build_y0(Eigen::MatrixXd x, int var_lag, int t);
RcppExport SEXP _bvhar_build_y0(SEXP xSEXP, SEXP var_lagSEXP, SEXP tSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type var_lag(var_lagSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    rcpp_result_gen = Rcpp::wrap(build_y0(x, var_lag, t));
    return rcpp_result_gen;
END_RCPP
}
// build_design
Eigen::MatrixXd build_design(Eigen::MatrixXd x, int p);
RcppExport SEXP _bvhar_build_design(SEXP xSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(build_design(x, p));
    return rcpp_result_gen;
END_RCPP
}
// diag_misc
Eigen::MatrixXd diag_misc(Eigen::VectorXd x);
RcppExport SEXP _bvhar_diag_misc(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(diag_misc(x));
    return rcpp_result_gen;
END_RCPP
}
// build_ydummy
Eigen::MatrixXd build_ydummy(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd delta);
RcppExport SEXP _bvhar_build_ydummy(SEXP pSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP deltaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type delta(deltaSEXP);
    rcpp_result_gen = Rcpp::wrap(build_ydummy(p, sigma, lambda, delta));
    return rcpp_result_gen;
END_RCPP
}
// build_xdummy
Eigen::MatrixXd build_xdummy(int p, double lambda, Eigen::VectorXd sigma, double eps);
RcppExport SEXP _bvhar_build_xdummy(SEXP pSEXP, SEXP lambdaSEXP, SEXP sigmaSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(build_xdummy(p, lambda, sigma, eps));
    return rcpp_result_gen;
END_RCPP
}
// minnesota_prior
Rcpp::List minnesota_prior(Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy);
RcppExport SEXP _bvhar_minnesota_prior(SEXP x_dummySEXP, SEXP y_dummySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x_dummy(x_dummySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y_dummy(y_dummySEXP);
    rcpp_result_gen = Rcpp::wrap(minnesota_prior(x_dummy, y_dummy));
    return rcpp_result_gen;
END_RCPP
}
// build_ydummy_bvhar
Eigen::MatrixXd build_ydummy_bvhar(Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly);
RcppExport SEXP _bvhar_build_ydummy_bvhar(SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP dailySEXP, SEXP weeklySEXP, SEXP monthlySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type daily(dailySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type weekly(weeklySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type monthly(monthlySEXP);
    rcpp_result_gen = Rcpp::wrap(build_ydummy_bvhar(sigma, lambda, daily, weekly, monthly));
    return rcpp_result_gen;
END_RCPP
}
// estimate_bvar_mn
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy);
RcppExport SEXP _bvhar_estimate_bvar_mn(SEXP xSEXP, SEXP ySEXP, SEXP x_dummySEXP, SEXP y_dummySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x_dummy(x_dummySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y_dummy(y_dummySEXP);
    rcpp_result_gen = Rcpp::wrap(estimate_bvar_mn(x, y, x_dummy, y_dummy));
    return rcpp_result_gen;
END_RCPP
}
// estimate_mn_flat
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U);
RcppExport SEXP _bvhar_estimate_mn_flat(SEXP xSEXP, SEXP ySEXP, SEXP USEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type U(USEXP);
    rcpp_result_gen = Rcpp::wrap(estimate_mn_flat(x, y, U));
    return rcpp_result_gen;
END_RCPP
}
// estimate_var
Rcpp::List estimate_var(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_estimate_var(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(estimate_var(x, y));
    return rcpp_result_gen;
END_RCPP
}
// compute_cov
Eigen::MatrixXd compute_cov(Eigen::MatrixXd z, int num_design, int dim_design);
RcppExport SEXP _bvhar_compute_cov(SEXP zSEXP, SEXP num_designSEXP, SEXP dim_designSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type num_design(num_designSEXP);
    Rcpp::traits::input_parameter< int >::type dim_design(dim_designSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_cov(z, num_design, dim_design));
    return rcpp_result_gen;
END_RCPP
}
// VARcoeftoVMA
Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);
RcppExport SEXP _bvhar_VARcoeftoVMA(SEXP var_coefSEXP, SEXP var_lagSEXP, SEXP lag_maxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type var_coef(var_coefSEXP);
    Rcpp::traits::input_parameter< int >::type var_lag(var_lagSEXP);
    Rcpp::traits::input_parameter< int >::type lag_max(lag_maxSEXP);
    rcpp_result_gen = Rcpp::wrap(VARcoeftoVMA(var_coef, var_lag, lag_max));
    return rcpp_result_gen;
END_RCPP
}
// VARtoVMA
Eigen::MatrixXd VARtoVMA(Rcpp::List object, int lag_max);
RcppExport SEXP _bvhar_VARtoVMA(SEXP objectSEXP, SEXP lag_maxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type lag_max(lag_maxSEXP);
    rcpp_result_gen = Rcpp::wrap(VARtoVMA(object, lag_max));
    return rcpp_result_gen;
END_RCPP
}
// compute_covmse
Eigen::MatrixXd compute_covmse(Rcpp::List object, int step);
RcppExport SEXP _bvhar_compute_covmse(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_covmse(object, step));
    return rcpp_result_gen;
END_RCPP
}
// scale_har
Eigen::MatrixXd scale_har(int m);
RcppExport SEXP _bvhar_scale_har(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(scale_har(m));
    return rcpp_result_gen;
END_RCPP
}
// estimate_har
Rcpp::List estimate_har(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_estimate_har(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(estimate_har(x, y));
    return rcpp_result_gen;
END_RCPP
}
// VHARcoeftoVMA
Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max);
RcppExport SEXP _bvhar_VHARcoeftoVMA(SEXP vhar_coefSEXP, SEXP HARtrans_matSEXP, SEXP lag_maxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type vhar_coef(vhar_coefSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type HARtrans_mat(HARtrans_matSEXP);
    Rcpp::traits::input_parameter< int >::type lag_max(lag_maxSEXP);
    rcpp_result_gen = Rcpp::wrap(VHARcoeftoVMA(vhar_coef, HARtrans_mat, lag_max));
    return rcpp_result_gen;
END_RCPP
}
// VHARtoVMA
Eigen::MatrixXd VHARtoVMA(Rcpp::List object, int lag_max);
RcppExport SEXP _bvhar_VHARtoVMA(SEXP objectSEXP, SEXP lag_maxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type lag_max(lag_maxSEXP);
    rcpp_result_gen = Rcpp::wrap(VHARtoVMA(object, lag_max));
    return rcpp_result_gen;
END_RCPP
}
// compute_covmse_har
Eigen::MatrixXd compute_covmse_har(Rcpp::List object, int step);
RcppExport SEXP _bvhar_compute_covmse_har(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_covmse_har(object, step));
    return rcpp_result_gen;
END_RCPP
}
// forecast_bvarmn
Rcpp::List forecast_bvarmn(Rcpp::List object, int step);
RcppExport SEXP _bvhar_forecast_bvarmn(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(forecast_bvarmn(object, step));
    return rcpp_result_gen;
END_RCPP
}
// forecast_bvarmn_flat
Rcpp::List forecast_bvarmn_flat(Rcpp::List object, int step);
RcppExport SEXP _bvhar_forecast_bvarmn_flat(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(forecast_bvarmn_flat(object, step));
    return rcpp_result_gen;
END_RCPP
}
// forecast_bvharmn
Rcpp::List forecast_bvharmn(Rcpp::List object, int step);
RcppExport SEXP _bvhar_forecast_bvharmn(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(forecast_bvharmn(object, step));
    return rcpp_result_gen;
END_RCPP
}
// forecast_var
SEXP forecast_var(Rcpp::List object, int step);
RcppExport SEXP _bvhar_forecast_var(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(forecast_var(object, step));
    return rcpp_result_gen;
END_RCPP
}
// forecast_vhar
SEXP forecast_vhar(Rcpp::List object, int step);
RcppExport SEXP _bvhar_forecast_vhar(SEXP objectSEXP, SEXP stepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type object(objectSEXP);
    Rcpp::traits::input_parameter< int >::type step(stepSEXP);
    rcpp_result_gen = Rcpp::wrap(forecast_vhar(object, step));
    return rcpp_result_gen;
END_RCPP
}
// AAt_eigen
Eigen::MatrixXd AAt_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_AAt_eigen(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(AAt_eigen(x, y));
    return rcpp_result_gen;
END_RCPP
}
// tAA_eigen
Eigen::MatrixXd tAA_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_tAA_eigen(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(tAA_eigen(x, y));
    return rcpp_result_gen;
END_RCPP
}
// AtAit_eigen
Eigen::MatrixXd AtAit_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_AtAit_eigen(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(AtAit_eigen(x, y));
    return rcpp_result_gen;
END_RCPP
}
// kroneckerprod
Eigen::MatrixXd kroneckerprod(Eigen::MatrixXd x, Eigen::MatrixXd y);
RcppExport SEXP _bvhar_kroneckerprod(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(kroneckerprod(x, y));
    return rcpp_result_gen;
END_RCPP
}
// sim_mgaussian
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);
RcppExport SEXP _bvhar_sim_mgaussian(SEXP num_simSEXP, SEXP muSEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_sim(num_simSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_mgaussian(num_sim, mu, sig));
    return rcpp_result_gen;
END_RCPP
}
// sim_var
Eigen::MatrixXd sim_var(int num_sim, int num_burn, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error, Eigen::MatrixXd init);
RcppExport SEXP _bvhar_sim_var(SEXP num_simSEXP, SEXP num_burnSEXP, SEXP var_coefSEXP, SEXP var_lagSEXP, SEXP sig_errorSEXP, SEXP initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_sim(num_simSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type var_coef(var_coefSEXP);
    Rcpp::traits::input_parameter< int >::type var_lag(var_lagSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sig_error(sig_errorSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type init(initSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_var(num_sim, num_burn, var_coef, var_lag, sig_error, init));
    return rcpp_result_gen;
END_RCPP
}
// sim_stable_var
Eigen::MatrixXd sim_stable_var(int num_sim, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error);
RcppExport SEXP _bvhar_sim_stable_var(SEXP num_simSEXP, SEXP var_coefSEXP, SEXP var_lagSEXP, SEXP sig_errorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_sim(num_simSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type var_coef(var_coefSEXP);
    Rcpp::traits::input_parameter< int >::type var_lag(var_lagSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sig_error(sig_errorSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_stable_var(num_sim, var_coef, var_lag, sig_error));
    return rcpp_result_gen;
END_RCPP
}
// sim_vhar
Eigen::MatrixXd sim_vhar(int num_sim, int num_burn, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error, Eigen::MatrixXd init);
RcppExport SEXP _bvhar_sim_vhar(SEXP num_simSEXP, SEXP num_burnSEXP, SEXP vhar_coefSEXP, SEXP sig_errorSEXP, SEXP initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_sim(num_simSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type vhar_coef(vhar_coefSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sig_error(sig_errorSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type init(initSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_vhar(num_sim, num_burn, vhar_coef, sig_error, init));
    return rcpp_result_gen;
END_RCPP
}
// sim_stable_vhar
Eigen::MatrixXd sim_stable_vhar(int num_sim, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error);
RcppExport SEXP _bvhar_sim_stable_vhar(SEXP num_simSEXP, SEXP vhar_coefSEXP, SEXP sig_errorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_sim(num_simSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type vhar_coef(vhar_coefSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sig_error(sig_errorSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_stable_vhar(num_sim, vhar_coef, sig_error));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bvhar_build_y0", (DL_FUNC) &_bvhar_build_y0, 3},
    {"_bvhar_build_design", (DL_FUNC) &_bvhar_build_design, 2},
    {"_bvhar_diag_misc", (DL_FUNC) &_bvhar_diag_misc, 1},
    {"_bvhar_build_ydummy", (DL_FUNC) &_bvhar_build_ydummy, 4},
    {"_bvhar_build_xdummy", (DL_FUNC) &_bvhar_build_xdummy, 4},
    {"_bvhar_minnesota_prior", (DL_FUNC) &_bvhar_minnesota_prior, 2},
    {"_bvhar_build_ydummy_bvhar", (DL_FUNC) &_bvhar_build_ydummy_bvhar, 5},
    {"_bvhar_estimate_bvar_mn", (DL_FUNC) &_bvhar_estimate_bvar_mn, 4},
    {"_bvhar_estimate_mn_flat", (DL_FUNC) &_bvhar_estimate_mn_flat, 3},
    {"_bvhar_estimate_var", (DL_FUNC) &_bvhar_estimate_var, 2},
    {"_bvhar_compute_cov", (DL_FUNC) &_bvhar_compute_cov, 3},
    {"_bvhar_VARcoeftoVMA", (DL_FUNC) &_bvhar_VARcoeftoVMA, 3},
    {"_bvhar_VARtoVMA", (DL_FUNC) &_bvhar_VARtoVMA, 2},
    {"_bvhar_compute_covmse", (DL_FUNC) &_bvhar_compute_covmse, 2},
    {"_bvhar_scale_har", (DL_FUNC) &_bvhar_scale_har, 1},
    {"_bvhar_estimate_har", (DL_FUNC) &_bvhar_estimate_har, 2},
    {"_bvhar_VHARcoeftoVMA", (DL_FUNC) &_bvhar_VHARcoeftoVMA, 3},
    {"_bvhar_VHARtoVMA", (DL_FUNC) &_bvhar_VHARtoVMA, 2},
    {"_bvhar_compute_covmse_har", (DL_FUNC) &_bvhar_compute_covmse_har, 2},
    {"_bvhar_forecast_bvarmn", (DL_FUNC) &_bvhar_forecast_bvarmn, 2},
    {"_bvhar_forecast_bvarmn_flat", (DL_FUNC) &_bvhar_forecast_bvarmn_flat, 2},
    {"_bvhar_forecast_bvharmn", (DL_FUNC) &_bvhar_forecast_bvharmn, 2},
    {"_bvhar_forecast_var", (DL_FUNC) &_bvhar_forecast_var, 2},
    {"_bvhar_forecast_vhar", (DL_FUNC) &_bvhar_forecast_vhar, 2},
    {"_bvhar_AAt_eigen", (DL_FUNC) &_bvhar_AAt_eigen, 2},
    {"_bvhar_tAA_eigen", (DL_FUNC) &_bvhar_tAA_eigen, 2},
    {"_bvhar_AtAit_eigen", (DL_FUNC) &_bvhar_AtAit_eigen, 2},
    {"_bvhar_kroneckerprod", (DL_FUNC) &_bvhar_kroneckerprod, 2},
    {"_bvhar_sim_mgaussian", (DL_FUNC) &_bvhar_sim_mgaussian, 3},
    {"_bvhar_sim_var", (DL_FUNC) &_bvhar_sim_var, 6},
    {"_bvhar_sim_stable_var", (DL_FUNC) &_bvhar_sim_stable_var, 4},
    {"_bvhar_sim_vhar", (DL_FUNC) &_bvhar_sim_vhar, 5},
    {"_bvhar_sim_stable_vhar", (DL_FUNC) &_bvhar_sim_stable_vhar, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_bvhar(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
