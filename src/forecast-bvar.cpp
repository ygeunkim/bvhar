#include "bvharomp.h"
#include "bvhardraw.h"

//' Forecasting BVAR(p)
//' 
//' @param object `bvarmn` or `bvarflat` object
//' @param step Integer, Step to forecast
//' @param num_sim Integer, number to simulate parameters from posterior distribution
//' @details
//' n-step ahead forecasting using BVAR(p) recursively.
//' 
//' For given number of simulation (`num_sim`),
//' 
//' 1. Generate \eqn{(A^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
//' 2. Recursively, \eqn{j = 1, \ldots, h} (`step`)
//'     - Point forecast: Use \eqn{\hat{A}}
//'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim A^{(b)}, \Sigma_e^{(b)} \sim MN}
//'     - tilde notation indicates simulated ones
//' 
//' @references
//' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' 
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897. doi:[10.1016/b978-0-444-62731-5.00015-4](https://doi.org/10.1016/B978-0-444-62731-5.00015-4)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvar(Rcpp::List object, int step, int num_sim) {
  if (!object.inherits("bvarmn") && !object.inherits("bvarflat")) {
    Rcpp::stop("'object' must be bvarmn or bvarflat object.");
  }
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["coefficients"]; // Ahat = posterior mean of MN
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // vhat = posterior precision of MN to compute SE
  Eigen::MatrixXd posterior_mn_scale_u = posterior_prec_mat.inverse();
  Eigen::MatrixXd posterior_scale = object["iw_scale"]; // Sighat = posterior scale of IW
  double posterior_shape = object["iw_shape"]; // posterior shape of IW
  int dim = object["m"]; // dimension of time series
  int var_lag = object["p"]; // VAR(p)
  int num_design = object["obs"]; // s = n - p
  int dim_design = object["df"];
  // (A, Sig) ~ MNIW
  Rcpp::List coef_and_sig = sim_mniw(
    num_sim, 
    posterior_mean_mat, 
    Eigen::Map<Eigen::MatrixXd>(posterior_mn_scale_u.data(), dim_design, dim_design), 
    Eigen::Map<Eigen::MatrixXd>(posterior_scale.data(), dim, dim), 
    posterior_shape
  );
  Eigen::MatrixXd coef_gen = coef_and_sig["mn"]; // generated Ahat: k x Bm
  Eigen::MatrixXd sig_gen = coef_and_sig["iw"]; // generated Sighat: m x Bm
  // forecasting step
  Eigen::MatrixXd point_forecast(step, dim); // h x m matrix
  Eigen::MatrixXd density_forecast(step, num_sim * dim); // h x Bm matrix
  Eigen::MatrixXd predictive_distn(step, num_sim * dim); // h x Bm matrix
  Eigen::MatrixXd last_pvec(1, dim_design); // vectorize the last p observation and include 1
  Eigen::MatrixXd tmp_vec(1, (var_lag - 1) * dim);
  Eigen::VectorXd sig_closed(step); // se^2 for each forecast (except Sigma2 part, i.e. closed form)
  for (int i = 0; i < step; i++) {
    sig_closed(i) = 1.0;
  }
  last_pvec(0, dim_design - 1) = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  sig_closed.block(0, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.transpose();
  point_forecast.block(0, 0, 1, dim) = last_pvec * posterior_mean_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Ahat
  density_forecast.block(0, 0, 1, num_sim * dim) = last_pvec * coef_gen; // use A(simulated)
  // one-step ahead forecasting
  Eigen::MatrixXd sig_mat = sig_gen.block(0, 0, dim, dim); // First Sighat
  for (int b = 0; b < num_sim; b++) {
    predictive_distn.block(0, b * dim, 1, dim) = sim_matgaussian(
      density_forecast.block(0, b * dim, 1, dim),
      Eigen::Map<Eigen::MatrixXd>(sig_closed.block(0, 0, 1, 1).data(), 1, 1),
      Eigen::Map<Eigen::MatrixXd>(sig_mat.data(), dim, dim)
    );
  }
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = point_forecast,
      Rcpp::Named("predictive") = predictive_distn
    );
  }
  // Next h - 1: recursively
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.block(0, 0, 1, (var_lag - 1) * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = point_forecast.block(i - 1, 0, 1, dim);
    sig_closed.block(i, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.transpose();
    point_forecast.block(i, 0, 1, dim) = last_pvec * posterior_mean_mat; // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Ahat
    // Predictive distribution
    density_forecast.block(i, 0, 1, num_sim * dim) = last_pvec * coef_gen;
    for (int b = 0; b < num_sim; b++) {
      sig_mat = sig_gen.block(0, b * dim, dim, dim); // b-th Sighat
      predictive_distn.block(i, b * dim, 1, dim) = sim_matgaussian(
        density_forecast.block(i, b * dim, 1, dim),
        Eigen::Map<Eigen::MatrixXd>(sig_closed.block(0, 0, 1, 1).data(), 1, 1),
        Eigen::Map<Eigen::MatrixXd>(sig_mat.data(), dim, dim)
      );
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = point_forecast,
    Rcpp::Named("predictive") = predictive_distn
  );
}

//' Forecasting VAR(p) with SSVS
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean of SSVS.
//' @param alpha_record Matrix, MCMC trace of alpha.
//' @param eta_record Matrix, MCMC trace of eta.
//' @param psi_record Matrix, MCMC trace of psi.
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarssvs(int var_lag,
                             int step,
                             Eigen::MatrixXd response_mat,
                             Eigen::MatrixXd coef_mat,
                             Eigen::MatrixXd alpha_record,
                             Eigen::MatrixXd eta_record,
                             Eigen::MatrixXd psi_record) {
  int num_sim = alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_design = coef_mat.rows();
  Eigen::MatrixXd point_forecast(step, dim);
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step, num_sim * dim);
  Eigen::VectorXd last_pvec(dim_design);
  Eigen::VectorXd tmp_vec((var_lag - 1) * dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  point_forecast.row(0) = last_pvec.transpose() * coef_mat;
  Eigen::MatrixXd chol_factor(dim, dim);
  Eigen::MatrixXd sig_cycle(dim, dim);
  for (int b = 0; b < num_sim; b++) {
    density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
    chol_factor = bvhar::build_chol(psi_record.row(b), eta_record.row(b));
    sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
    predictive_distn.block(0, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
  }
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = point_forecast,
      Rcpp::Named("predictive") = predictive_distn
    );
  }
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.segment(0, (var_lag - 1) * dim);
    last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.segment(0, dim) = point_forecast.row(i - 1);
    point_forecast.row(i) = last_pvec.transpose() * coef_mat;
    for (int b = 0; b < num_sim; b++) {
      density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
      chol_factor = bvhar::build_chol(psi_record.row(b), eta_record.row(b));
      sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
      predictive_distn.block(i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = point_forecast,
    Rcpp::Named("predictive") = predictive_distn
  );
}

//' Forecasting VAR(p) with Horseshoe Prior
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean of SSVS.
//' @param alpha_record Matrix, MCMC trace of alpha.
//' @param eta_record Matrix, MCMC trace of eta.
//' @param omega_record Matrix, MCMC trace of omega.
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarhs(int var_lag,
                           int step,
                           Eigen::MatrixXd response_mat,
                           Eigen::MatrixXd coef_mat,
                           Eigen::MatrixXd alpha_record,
                           Eigen::MatrixXd eta_record,
                           Eigen::MatrixXd omega_record) {
  int num_sim = alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_design = coef_mat.rows();
  Eigen::MatrixXd point_forecast(step, dim);
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step, num_sim * dim);
  Eigen::VectorXd last_pvec(dim_design);
  Eigen::VectorXd tmp_vec((var_lag - 1) * dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  point_forecast.row(0) = last_pvec.transpose() * coef_mat;
  Eigen::MatrixXd sig_cycle(dim, dim);
  for (int b = 0; b < num_sim; b++) {
    density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
    sig_cycle = bvhar::build_cov(omega_record.row(b), eta_record.row(b));
    predictive_distn.block(0, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
  }
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = point_forecast,
      Rcpp::Named("predictive") = predictive_distn
    );
  }
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.segment(0, (var_lag - 1) * dim);
    last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.segment(0, dim) = point_forecast.row(i - 1);
    point_forecast.row(i) = last_pvec.transpose() * coef_mat;
    for (int b = 0; b < num_sim; b++) {
      density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
      sig_cycle = bvhar::build_cov(omega_record.row(b), eta_record.row(b));
      predictive_distn.block(i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = point_forecast,
    Rcpp::Named("predictive") = predictive_distn
  );
}

//' Forecasting VAR-SV
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_bvarsv(int var_lag, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd coef_mat) {
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_design = coef_mat.rows();
  Eigen::MatrixXd point_forecast(step, dim);
  Eigen::VectorXd last_pvec(dim_design);
  Eigen::VectorXd tmp_vec((var_lag - 1) * dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  point_forecast.row(0) = last_pvec.transpose() * coef_mat;
  if (step == 1) {
    return point_forecast;
  }
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.segment(0, (var_lag - 1) * dim);
    last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.segment(0, dim) = point_forecast.row(i - 1);
    point_forecast.row(i) = last_pvec.transpose() * coef_mat;
  }
  return point_forecast;
}

//' Forecasting predictive density of VAR-SV
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param alpha_record MCMC record of coefficients
//' @param h_last_record MCMC record of log-volatilities in last time
//' @param a_record MCMC record of contemporaneous coefficients
//' @param sigh_record MCMC record of variance of log-volatilities
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarsv_density(int var_lag,
                                   int step,
                                   Eigen::MatrixXd response_mat,
                                   Eigen::MatrixXd coef_mat,
                                   Eigen::MatrixXd alpha_record,
                                   Eigen::MatrixXd h_last_record,
                                   Eigen::MatrixXd a_record,
                                   Eigen::MatrixXd sigh_record,
																	 bool include_mean) {
  int num_sim = alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_design = coef_mat.rows();
  Eigen::MatrixXd point_forecast(step, dim);
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step, num_sim * dim);
  Eigen::VectorXd last_pvec(dim_design);
  Eigen::VectorXd tmp_vec((var_lag - 1) * dim);
  Eigen::VectorXd sv_update(dim);
  Eigen::MatrixXd sv_cov = Eigen::MatrixXd::Zero(dim, dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  point_forecast.row(0) = last_pvec.transpose() * coef_mat;
	Eigen::MatrixXd coef_mat_record(coef_mat.rows(), dim); // include constant term
	int num_coef = coef_mat.size();
	int num_alpha = include_mean ? num_coef - dim : num_coef;
  Eigen::MatrixXd contem_mat = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd tvp_lvol = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd tvp_prec(dim, dim);
  for (int b = 0; b < num_sim; b++) {
    // density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
		coef_mat_record.topRows(var_lag * dim) = bvhar::unvectorize(alpha_record.row(b).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat_record.bottomRows(1) = alpha_record.row(b).tail(dim);
		}
		density_forecast = last_pvec.transpose() * coef_mat_record;
    sv_cov.diagonal() = 1 / sigh_record.row(b).array(); // covariance of h_t
    sv_update = bvhar::vectorize_eigen(
      sim_mgaussian_chol(1, h_last_record.row(b), sv_cov)
    ); // h_T+1 = h_T + u_T
    tvp_lvol.diagonal() = 1 / sv_update.array().exp(); // Dt = diag(exp(h_t))
    contem_mat = bvhar::build_inv_lower(dim, a_record.row(b));
    tvp_prec = contem_mat.transpose() * tvp_lvol * contem_mat; // L^T D_T  L
    predictive_distn.block(0, b * dim, 1, dim) = sim_mgaussian_chol(
      1,
      density_forecast,
      tvp_prec.inverse()
    );
  }
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = point_forecast,
      Rcpp::Named("predictive") = predictive_distn
    );
  }
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.segment(0, (var_lag - 1) * dim);
    last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.segment(0, dim) = point_forecast.row(i - 1);
    point_forecast.row(i) = last_pvec.transpose() * coef_mat;
    for (int b = 0; b < num_sim; b++) {
      // density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_record.row(b), dim);
			coef_mat_record.topRows(var_lag * dim) = bvhar::unvectorize(alpha_record.row(b).head(num_alpha).transpose(), dim);
			if (include_mean) {
				coef_mat_record.bottomRows(1) = alpha_record.row(b).tail(dim);
			}
			density_forecast = last_pvec.transpose() * coef_mat_record;
      sv_cov.diagonal() = 1 / sigh_record.row(b).array();
      sv_update = bvhar::vectorize_eigen(
        sim_mgaussian_chol(1, h_last_record.row(b), sv_cov)
      );
      tvp_lvol.diagonal() = 1 / sv_update.array();
      contem_mat = bvhar::build_inv_lower(dim, a_record.row(b));
      tvp_prec = contem_mat.transpose() * tvp_lvol * contem_mat;
      predictive_distn.block(i, b * dim, 1, dim) = sim_mgaussian_chol(
        1,
        density_forecast,
        tvp_prec.inverse()
      );
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = point_forecast,
    Rcpp::Named("predictive") = predictive_distn
  );
}

//' Out-of-Sample Forecasting of BVAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvar(Eigen::MatrixXd y, 
                          int lag, 
                          Rcpp::List bayes_spec,
                          bool include_mean, 
                          int step,
                          Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of BVAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR with Flat prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvarflat(Eigen::MatrixXd y, 
                              int lag, 
                              Rcpp::List bayes_spec,
                              bool include_mean, 
                              int step,
                              Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_flat");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of BVAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvar(Eigen::MatrixXd y, 
                            int lag, 
                            Rcpp::List bayes_spec,
                            bool include_mean, 
                            int step,
                            Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvar_mod = fit(y, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      lag, 
      bayes_spec, 
      include_mean
    );
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of BVAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVAR with Flat prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvarflat(Eigen::MatrixXd y, 
                                int lag, 
                                Rcpp::List bayes_spec,
                                bool include_mean, 
                                int step,
                                Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_flat");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvar_mod = fit(y, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      lag, 
      bayes_spec, 
      include_mean
    );
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvarsv(Eigen::MatrixXd y, int lag, int num_iter, int num_burn, int thinning, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads_roll, int nthreads_mod) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_sv");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
  Eigen::MatrixXd roll_mat = y;
  Rcpp::List bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
  Eigen::MatrixXd y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
  // Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
#ifdef _OPENMP
  Eigen::MatrixXd tot_mat(window + num_test, dim); // entire data set = train + test for parallel
  tot_mat.topRows(window) = y;
  tot_mat.bottomRows(num_test) = y_test;
  // shared(res, num_horizon, tot_mat, window, dim, fit, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, step) \
#pragma omp parallel for num_threads(nthreads_roll) private(roll_mat, bvar_mod, y_pred)
  for (int i = 1; i < num_horizon; i++) {
    // roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    // roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    roll_mat = tot_mat.block(i, 0, window, dim);
    bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
    res.row(i) = y_pred.row(step - 1);
  }
#else
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
    res.row(i) = y_pred.row(step - 1);
  }
#endif
  return res;
}
