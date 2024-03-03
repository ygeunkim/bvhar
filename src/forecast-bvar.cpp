#include "bvhardraw.h"

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
			density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b).eval(), dim);
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
				density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b).eval(), dim);
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
			density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b).eval(), dim);
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
				density_forecast = last_pvec.transpose() * bvhar::unvectorize(alpha_chain.row(b).eval(), dim);
				sig_cycle.setIdentity();
				sig_cycle *= sig_chain[b];
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
	return predictive_distn;
}
