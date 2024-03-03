#include "bvhardraw.h"

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
			density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b).eval(), dim);
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
				density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b).eval(), dim);
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
			density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b).eval(), dim);
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
				density_forecast = last_pvec.transpose() * HARtrans.transpose() * bvhar::unvectorize(phi_chain.row(b).eval(), dim);
				sig_cycle.setIdentity();
				sig_cycle *= sig_chain[b];
				predictive_distn.block(chain * step + i, b * dim, 1, dim) = sim_mgaussian_chol(1, density_forecast, sig_cycle);
			}
		}
	}
	return predictive_distn;
}
