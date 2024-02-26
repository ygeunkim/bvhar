#include "svforecaster.h"
#include "bvharinterrupt.h"

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
Eigen::MatrixXd forecast_bvarssvs(int num_chains, int var_lag, int step,
                             			Eigen::MatrixXd response_mat,
																	int dim_design,
                             			Eigen::MatrixXd alpha_record,
																	Eigen::MatrixXd eta_record,
																	Eigen::MatrixXd psi_record) {
  int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
	int num_coef = dim_design * dim;
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step * num_chains, num_sim * dim);
  Eigen::VectorXd last_pvec(dim_design);
  Eigen::VectorXd tmp_vec(dim_design - dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  Eigen::MatrixXd chol_factor(dim, dim);
  Eigen::MatrixXd sig_cycle(dim, dim);
	Eigen::MatrixXd alpha_chain(num_sim, num_coef);
	Eigen::MatrixXd eta_chain(num_sim, dim * (dim - 1) / 2);
	Eigen::MatrixXd psi_chain(num_sim, dim);
	for (int chain = 0; chain < num_chains; chain++) {
		alpha_chain = alpha_record.middleRows(chain * num_sim, num_sim);
		eta_chain = eta_record.middleRows(chain * num_sim, num_sim);
		psi_chain = psi_record.middleRows(chain * num_sim, num_sim);
		for (int b = 0; b < num_sim; b++) {
			density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b), dim);
			chol_factor = bvhar::build_chol(psi_chain.row(b), eta_chain.row(b));
			sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
			predictive_distn.block(chain * step, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
		}
	}
  if (step == 1) {
    return predictive_distn;
  }
	for (int chain = 0; chain < num_chains; chain++) {
		alpha_chain = alpha_record.middleRows(chain * num_sim, num_sim);
		eta_chain = eta_record.middleRows(chain * num_sim, num_sim);
		psi_chain = psi_record.middleRows(chain * num_sim, num_sim);
		for (int i = 1; i < step; i++) {
			for (int b = 0; b < num_sim; b++) {
				tmp_vec = last_pvec.head(dim_design - dim);
				last_pvec << density_forecast, tmp_vec;
				density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b), dim);
				chol_factor = bvhar::build_chol(psi_chain.row(b), eta_chain.row(b));
				sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
	return predictive_distn;
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
Eigen::MatrixXd forecast_bvarhs(int num_chains, int var_lag, int step,
                           			Eigen::MatrixXd response_mat,
																int dim_design,
																Eigen::MatrixXd alpha_record,
																Eigen::VectorXd sigma_record) {
  int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
	int num_coef = dim_design * dim;
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step * num_chains, num_sim * dim);
  Eigen::VectorXd last_pvec(dim_design);
	Eigen::VectorXd tmp_vec(dim_design - dim);
  last_pvec[dim_design - 1] = 1.0;
  for (int i = 0; i < var_lag; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  Eigen::MatrixXd sig_cycle(dim, dim);
	Eigen::MatrixXd alpha_chain(num_sim, num_coef);
	Eigen::VectorXd sig_chain(num_sim);
	for (int chain = 0; chain < num_chains; chain++) {
		alpha_chain = alpha_record.middleRows(chain * num_sim, num_sim);
		sig_chain = sigma_record.segment(chain * num_sim, num_sim);
		for (int b = 0; b < num_sim; b++) {
			density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b), dim);
			sig_cycle.setIdentity();
			sig_cycle *= sig_chain[b];
			predictive_distn.block(chain * step, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
		}
	}
  if (step == 1) {
		return predictive_distn;
  }
	for (int chain = 0; chain < num_chains; chain++) {
		alpha_chain = alpha_record.middleRows(chain * num_sim, num_sim);
		sig_chain = sigma_record.segment(chain * num_sim, num_sim);
		for (int i = 1; i < step; i++) {
			for (int b = 0; b < num_sim; b++) {
				tmp_vec = last_pvec.head(dim_design - dim);
				last_pvec << density_forecast, tmp_vec;
				density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b), dim);
				sig_cycle.setIdentity();
				sig_cycle *= sig_chain[b];
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
	return predictive_distn;
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
Rcpp::List forecast_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
                           Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
													 Eigen::VectorXi seed_chain, bool include_mean) {
	int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVarForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			alpha_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i] = std::unique_ptr<bvhar::SvVarForecaster>(new bvhar::SvVarForecaster(
			sv_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
	}
	return Rcpp::wrap(res);
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
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											 Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads_roll, int nthreads_mod) {
  int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
	}
	std::vector<std::vector<Rcpp::List>> records(num_horizon, std::vector<Rcpp::List>(num_chains));
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					records[window][chain] = sv_objs[window][chain]->returnRecords(0, 1);
				}
				std::unique_ptr<bvhar::SvRecords> sv_record;
				if (include_mean) {
					sv_record.reset(new bvhar::SvRecords(
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
					));
				} else {
					sv_record.reset(new bvhar::SvRecords(
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
					));
				}
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					*sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			records[window][chain] = sv_objs[window][chain]->returnRecords(num_burn, thinning);
		}
		std::unique_ptr<bvhar::SvRecords> sv_record;
		if (include_mean) {
			sv_record.reset(new bvhar::SvRecords(
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
			));
		} else {
			sv_record.reset(new bvhar::SvRecords(
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
			));
		}
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			*sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
	};
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads_roll)
#endif
	for (int window = 0; window < num_horizon; window++) {
		if (num_chains == 1) {
			run_gibbs(window, 0);
			res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads_mod)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				run_gibbs(window, chain);
				res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
			}
		}
	}
  return Rcpp::wrap(res);
}
