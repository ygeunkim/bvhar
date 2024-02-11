#include "bvharomp.h"
#include "bvhardraw.h"

//' Forecasting Bayesian VHAR
//' 
//' @param object `bvharmn` object
//' @param step Integer, Step to forecast
//' @param num_sim Integer, number to simulate parameters from posterior distribution
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' For given number of simulation (`num_sim`),
//' 
//' 1. Generate \eqn{(\Phi^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
//' 2. Recursively, \eqn{j = 1, \ldots, h} (`step`)
//'     - Point forecast: Use \eqn{\hat\Phi}
//'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim \Phi^{(b)}, \Sigma_e^{(b)} \sim MN}
//'     - tilde notation indicates simulated ones
//' 
//' @references Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. submitted.
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharmn(Rcpp::List object, int step, int num_sim) {
  if (!object.inherits("bvharmn")) {
    Rcpp::stop("'object' must be bvharmn object.");
  }
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["coefficients"]; // Phihat = posterior mean of MN: h x m, h = 3m (+ 1)
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // Psihat = posterior precision of MN to compute SE: h x h
  Eigen::MatrixXd posterior_mn_scale_u = posterior_prec_mat.inverse();
  Eigen::MatrixXd posterior_scale = object["iw_scale"]; // Sighat = posterior scale of IW: m x m
  double posterior_shape = object["iw_shape"]; // posterior shape of IW
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation: h x k0, k0 = 22m (+ 1)
  Eigen::MatrixXd transformed_prec_mat = HARtrans.transpose() * posterior_prec_mat.inverse() * HARtrans; // to compute SE: play a role V in BVAR
  int dim = object["m"]; // dimension of time series
  int num_design = object["obs"]; // s = n - p
  int dim_design = object["df"]; // 3m + 1 (const) or 3m (none)
  int dim_har = HARtrans.cols(); // 22m + 1 (const) or 22m (none)
  int month = object["month"];
  // (Phi, Sig) ~ MNIW
  Rcpp::List coef_and_sig = sim_mniw(
    num_sim, 
    posterior_mean_mat, 
    Eigen::Map<Eigen::MatrixXd>(posterior_mn_scale_u.data(), dim_design, dim_design), 
    Eigen::Map<Eigen::MatrixXd>(posterior_scale.data(), dim, dim), 
    posterior_shape
  );
  Eigen::MatrixXd coef_gen = coef_and_sig["mn"]; // generated Phihat: h x Bm, h = 3m (+ 1)
  Eigen::MatrixXd sig_gen = coef_and_sig["iw"]; // generated Sighat: m x Bm
  // forecasting step
  Eigen::MatrixXd point_forecast(step, dim); // h x m matrix
  Eigen::MatrixXd density_forecast(step, num_sim * dim); // h x Bm matrix
  Eigen::MatrixXd predictive_distn(step, num_sim * dim); // h x Bm matrix
  Eigen::MatrixXd last_pvec(1, dim_har); // vectorize the last 22 observation and include 1
  Eigen::MatrixXd tmp_vec(1, (month - 1) * dim);
  Eigen::MatrixXd res(step, dim); // h x m matrix
  Eigen::VectorXd sig_closed(step); // se^2 for each forecast (except Sigma2 part, i.e. closed form)
  for (int i = 0; i < step; i++) {
    sig_closed(i) = 1.0;
  }
  last_pvec(0, dim_har - 1) = 1.0;
  for (int i = 0; i < month; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  sig_closed.block(0, 0, 1, 1) += last_pvec * transformed_prec_mat * last_pvec.transpose();
  point_forecast.block(0, 0, 1, dim) = last_pvec * HARtrans.transpose() * posterior_mean_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
  density_forecast.block(0, 0, 1, num_sim * dim) = last_pvec * HARtrans.transpose() * coef_gen; // (1, k0) x (k0, h) x (h, Bm) = (1, Bm)
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
    tmp_vec = last_pvec.block(0, 0, 1, (month - 1) * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, (month - 1) * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = point_forecast.block(i - 1, 0, 1, dim);
    sig_closed.block(i, 0, 1, 1) += last_pvec * transformed_prec_mat * last_pvec.transpose();
    // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
    point_forecast.block(i, 0, 1, dim) = last_pvec * HARtrans.transpose() * posterior_mean_mat;
    // Predictive distribution
    density_forecast.block(i, 0, 1, num_sim * dim) = last_pvec * HARtrans.transpose() * coef_gen;
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

//' Forecasting VHAR with SSVS
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean of SSVS.
//' @param HARtrans VHAR linear transformation matrix
//' @param phi_record Matrix, MCMC trace of alpha.
//' @param eta_record Matrix, MCMC trace of eta.
//' @param psi_record Matrix, MCMC trace of psi.
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_bvharssvs(int num_chains, int month, int step,
                              		 Eigen::MatrixXd response_mat,
																	 Eigen::MatrixXd HARtrans,
																	 Eigen::MatrixXd phi_record,
																	 Eigen::MatrixXd eta_record,
																	 Eigen::MatrixXd psi_record) {
  int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int lag_var = HARtrans.cols();
  int dim_har = HARtrans.rows();
	int num_coef = dim_har * dim;
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step * num_chains, num_sim * dim);
  Eigen::VectorXd last_pvec(lag_var);
	Eigen::VectorXd tmp_vec(lag_var - dim);
  last_pvec[lag_var - 1] = 1.0;
  for (int i = 0; i < month; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  Eigen::MatrixXd chol_factor(dim, dim);
  Eigen::MatrixXd sig_cycle(dim, dim);
	Eigen::MatrixXd phi_chain(num_sim, num_coef);
	Eigen::MatrixXd eta_chain(num_sim, dim * (dim - 1) / 2);
	Eigen::MatrixXd psi_chain(num_sim, dim);
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		eta_chain = eta_record.middleRows(chain * num_sim, num_sim);
		psi_chain = psi_record.middleRows(chain * num_sim, num_sim);
		for (int b = 0; b < num_sim; b++) {
			density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b), dim);
			chol_factor = bvhar::build_chol(psi_chain.row(b), eta_chain.row(b));
			sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
			predictive_distn.block(chain * step, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
		}
	}
  if (step == 1) {
    return predictive_distn;
  }
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		eta_chain = eta_record.middleRows(chain * num_sim, num_sim);
		psi_chain = psi_record.middleRows(chain * num_sim, num_sim);
		for (int i = 1; i < step; i++) {
			for (int b = 0; b < num_sim; b++) {
				tmp_vec = last_pvec.head(lag_var - dim);
				last_pvec << density_forecast, tmp_vec;
				density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b), dim);
				chol_factor = bvhar::build_chol(psi_chain.row(b), eta_chain.row(b));
				sig_cycle = (chol_factor * chol_factor.transpose()).inverse();
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
  return predictive_distn;
}

//' Forecasting VHAR with Horseshoe Prior
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean of SSVS.
//' @param HARtrans VHAR linear transformation matrix
//' @param phi_record Matrix, MCMC trace of phi.
//' @param eta_record Matrix, MCMC trace of eta.
//' @param omega_record Matrix, MCMC trace of omega.
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_bvharhs(int num_chains, int month, int step,
																 Eigen::MatrixXd response_mat,
																 Eigen::MatrixXd HARtrans,
																 Eigen::MatrixXd phi_record,
																 Eigen::VectorXd sigma_record) {
  int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int lag_var = HARtrans.cols();
  int dim_har = HARtrans.rows();
	int num_coef = dim_har * dim;
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step * num_chains, num_sim * dim);
  Eigen::VectorXd last_pvec(lag_var);
	Eigen::VectorXd tmp_vec(lag_var - dim);
  last_pvec[lag_var - 1] = 1.0;
  for (int i = 0; i < month; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  Eigen::MatrixXd sig_cycle(dim, dim);
	Eigen::MatrixXd phi_chain(num_sim, num_coef);
	Eigen::VectorXd sig_chain(num_sim);
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		sig_chain = sigma_record.segment(chain * num_sim, num_sim);
		for (int b = 0; b < num_sim; b++) {
			density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b), dim);
			sig_cycle.setIdentity();
			sig_cycle *= sig_chain[b];
			predictive_distn.block(chain * step, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
		}
	}
  if (step == 1) {
		return predictive_distn;
  }
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		sig_chain = sigma_record.segment(chain * num_sim, num_sim);
		for (int i = 1; i < step; i++) {
			for (int b = 0; b < num_sim; b++) {
				tmp_vec = last_pvec.head(lag_var - dim);
				last_pvec << density_forecast, tmp_vec;
				density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b), dim);
				sig_cycle.setIdentity();
				sig_cycle *= sig_chain[b];
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
	return predictive_distn;
}

//' Forecasting VHAR-SV
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param HARtrans VHAR linear transformation matrix
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_bvharsv(int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd coef_mat, Eigen::MatrixXd HARtrans) {
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int dim_har = HARtrans.cols();
  Eigen::MatrixXd point_forecast(step, dim);
  Eigen::VectorXd last_pvec(dim_har);
  Eigen::VectorXd tmp_vec((month - 1) * dim);
  last_pvec[dim_har - 1] = 1.0;
  for (int i = 0; i < month; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
  point_forecast.row(0) = last_pvec.transpose() * HARtrans.transpose() * coef_mat;
  if (step == 1) {
    return point_forecast;
  }
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.segment(0, (month - 1) * dim);
    last_pvec.segment(dim, (month - 1) * dim) = tmp_vec;
    last_pvec.segment(0, dim) = point_forecast.row(i - 1);
    point_forecast.row(i) = last_pvec.transpose() * HARtrans.transpose() * coef_mat;
  }
  return point_forecast;
}

//' Forecasting Predictive Density of VHAR-SV
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param HARtrans VHAR linear transformation matrix
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_bvharsv_density(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
																 				 Eigen::MatrixXd phi_record, Eigen::MatrixXd h_last_record,
																				 Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record, bool include_mean) {
	int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
  int dim = response_mat.cols();
  int num_design = response_mat.rows();
  int lag_var = HARtrans.cols(); // month * dim( + 1)
  int dim_har = HARtrans.rows(); // 3 * dim( + 1)
	int num_coef = dim_har * dim;
	int num_alpha = include_mean ? num_coef - dim : num_coef;
  Eigen::VectorXd density_forecast(dim);
  Eigen::MatrixXd predictive_distn(step * num_chains, num_sim * dim);
  Eigen::VectorXd last_pvec(lag_var);
	Eigen::VectorXd tmp_vec(lag_var - dim); // (month - 1) * dim (+ 1)
  Eigen::VectorXd sv_update(dim);
  Eigen::MatrixXd sv_cov = Eigen::MatrixXd::Zero(dim, dim);
  last_pvec[lag_var - 1] = 1.0;
  for (int i = 0; i < month; i++) {
    last_pvec.segment(i * dim, dim) = response_mat.row(num_design - 1 - i);
  }
	Eigen::MatrixXd coef_mat_record(dim_har, dim); // include constant term
  Eigen::MatrixXd contem_mat = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd tvp_lvol = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd tvp_prec(dim, dim);
	Eigen::MatrixXd phi_chain(num_sim, num_coef);
	Eigen::MatrixXd h_chain(num_sim, dim);
	Eigen::MatrixXd a_chain(num_sim, dim * (dim - 1) / 2);
	Eigen::MatrixXd sigh_chain(num_sim, dim);
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		h_chain = h_last_record.middleRows(chain * num_sim, num_sim);
		a_chain = a_record.middleRows(chain * num_sim, num_sim);
		sigh_chain = sigh_record.middleRows(chain * num_sim, num_sim);
		for (int b = 0; b < num_sim; b++) {
			coef_mat_record.topRows(3 * dim) = bvhar::unvectorize(phi_chain.row(b).head(num_alpha).transpose(), dim);
			if (include_mean) {
				coef_mat_record.bottomRows(1) = phi_chain.row(b).tail(dim);
			}
			density_forecast = last_pvec.transpose() * HARtrans.transpose() * coef_mat_record;
			sv_cov.diagonal() = 1 / sigh_chain.row(b).array(); // covariance of h_t
			sv_update = bvhar::vectorize_eigen(
				sim_mgaussian_chol(1, h_chain.row(b), sv_cov)
			); // h_T+1 = h_T + u_T
			tvp_lvol.diagonal() = 1 / sv_update.array();
			contem_mat = bvhar::build_inv_lower(dim, a_chain.row(b));
			tvp_prec = contem_mat.transpose() * tvp_lvol * contem_mat; // L^T D_T  L
			predictive_distn.block(chain * step, b * dim, 1, dim) = sim_mgaussian_chol(
				1, density_forecast, tvp_prec.inverse()
			);
		}
	}
  if (step == 1) {
		return predictive_distn;
  }
	for (int chain = 0; chain < num_chains; chain++) {
		phi_chain = phi_record.middleRows(chain * num_sim, num_sim);
		h_chain = h_last_record.middleRows(chain * num_sim, num_sim);
		a_chain = a_record.middleRows(chain * num_sim, num_sim);
		sigh_chain = sigh_record.middleRows(chain * num_sim, num_sim);
		for (int i = 1; i < step; i++) {
			for (int b = 0; b < num_sim; b++) {
				tmp_vec = last_pvec.head(lag_var - dim);
				last_pvec << density_forecast, tmp_vec;
				coef_mat_record.topRows(3 * dim) = bvhar::unvectorize(phi_chain.row(b).head(num_alpha).transpose(), dim);
				if (include_mean) {
					coef_mat_record.bottomRows(1) = phi_chain.row(b).tail(dim);
				}
				density_forecast = last_pvec.transpose() * HARtrans.transpose() * coef_mat_record;
				sv_cov.diagonal() = 1 / sigh_chain.row(b).array(); // covariance of h_t
				sv_update = bvhar::vectorize_eigen(
					sim_mgaussian_chol(1, h_chain.row(b), sv_cov)
				); // h_T+1 = h_T + u_T
				tvp_lvol.diagonal() = 1 / sv_update.array().exp();
				contem_mat = bvhar::build_inv_lower(dim, a_chain.row(b));
				tvp_prec = contem_mat.transpose() * tvp_lvol * contem_mat; // L^T D_T  L
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(
					1,
					density_forecast,
					tvp_prec.inverse()
				);
			}
		}
	}
	return predictive_distn; // rbind(chains), cbind(sims)
}

//' Out-of-Sample Forecasting of BVHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvhar(Eigen::MatrixXd y, 
                           Eigen::VectorXd har,
                           Rcpp::List bayes_spec,
                           bool include_mean, 
                           int step,
                           Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvhar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvhar_mod = fit(roll_mat, har, bayes_spec, include_mean);
  Rcpp::List bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvhar_mod = fit(roll_mat, har, bayes_spec, include_mean);
    bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
    y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of VHAR-SV based on Rolling Window
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
Eigen::MatrixXd roll_bvharsv(Eigen::MatrixXd y, Eigen::VectorXi har,
                             int num_iter, int num_burn, int thinning,
                             Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
                             int nthreads_roll, int nthreads_mod) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvhar_sv");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
  Eigen::MatrixXd roll_mat = y;
  Rcpp::List bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
  Eigen::MatrixXd y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
  // Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
#ifdef _OPENMP
  Eigen::MatrixXd tot_mat(window + num_test, dim); // entire data set = train + test for parallel
  tot_mat.topRows(window) = y;
  tot_mat.bottomRows(num_test) = y_test;
#pragma omp parallel for num_threads(nthreads_roll) private(roll_mat, bvhar_mod, y_pred)
  for (int i = 1; i < num_horizon; i++) {
    // roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    // roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    roll_mat = tot_mat.block(i, 0, window, dim);
    bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
    // y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
#else
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
    // y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
#endif
  return res;
}

//' Out-of-Sample Forecasting of BVHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvhar(Eigen::MatrixXd y, 
                             Eigen::VectorXd har,
                             Rcpp::List bayes_spec,
                             bool include_mean, 
                             int step,
                             Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvhar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvhar_mod = fit(y, har, bayes_spec, include_mean);
  Rcpp::List bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvhar_mod = fit(
      expand_mat.block(0, 0, window + i, dim), 
      har,
      bayes_spec, 
      include_mean
    );
    bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
    y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}
