#include "bvharomp.h"
#include "regspillover.h"

// [[Rcpp::export]]
Rcpp::List compute_varldlt_spillover(int lag, int step,
																		 Eigen::MatrixXd alpha_record, Eigen::MatrixXd d_record, Eigen::MatrixXd a_record) {
	bvhar::LdltRecords reg_record(alpha_record, a_record, d_record);
	std::unique_ptr<bvhar::RegSpillover> spillover;
	spillover.reset(new bvhar::RegSpillover(reg_record, step, lag));
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}

// [[Rcpp::export]]
Rcpp::List compute_vharldlt_spillover(int week, int month, int step,
																			Eigen::MatrixXd phi_record, Eigen::MatrixXd d_record, Eigen::MatrixXd a_record) {
	int dim = d_record.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, false);
	bvhar::LdltRecords reg_record(phi_record, a_record, d_record);
	std::unique_ptr<bvhar::RegSpillover> spillover;
	spillover.reset(new bvhar::RegVharSpillover(reg_record, step, month, har_trans));
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvarldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																			int lag, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			int prior_type, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
	int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rf_error("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::McmcReg>> sur_objs(num_horizon);
	std::vector<std::unique_ptr<bvhar::RegSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, lag, lag + 1);
		Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, lag, include_mean);
		// Rcpp::List init_spec = param_init[i];
		Rcpp::List init_spec = param_init;
		switch (prior_type) {
			case 1: {
				bvhar::MinnParams minn_params(
					num_iter, roll_x0, roll_y0,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				bvhar::LdltInits ldlt_inits(init_spec);
				sur_objs[i].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 2: {
				bvhar::SsvsParams ssvs_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				bvhar::SsvsInits ssvs_inits(init_spec);
				sur_objs[i].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 3: {
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				bvhar::HsInits hs_inits(init_spec);
				sur_objs[i].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 4: {
				bvhar::HierminnParams minn_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				bvhar::HierminnInits minn_inits(init_spec);
				sur_objs[i].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
		}
	}
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; ++window) {
		for (int i = 0; i < num_iter; ++i) {
			sur_objs[window]->doPosteriorDraws();
		}
		bvhar::LdltRecords reg_record = sur_objs[window]->returnLdltRecords(num_burn, thin);
		spillover[window].reset(new bvhar::RegSpillover(reg_record, step, lag));
		spillover[window]->computeSpillover();
		to_sp.row(window) = spillover[window]->returnTo();
		from_sp.row(window) = spillover[window]->returnFrom();
		tot[window] = spillover[window]->returnTot();
		sur_objs[window].reset();
		spillover[window].reset();
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvharldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																			 int week, int month, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			 int prior_type, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			 bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
	int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rf_error("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::McmcReg>> sur_objs(num_horizon);
	std::vector<std::unique_ptr<bvhar::RegSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
	Eigen::MatrixXd har_trans = bvhar::build_vhar(y.cols(), week, month, include_mean);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, month, month + 1);
		Eigen::MatrixXd roll_x1 = bvhar::build_x0(roll_mat, month, include_mean) * har_trans.transpose();
		// Rcpp::List init_spec = param_init[i];
		Rcpp::List init_spec = param_init;
		switch (prior_type) {
			case 1: {
				bvhar::MinnParams minn_params(
					num_iter, roll_x1, roll_y0,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				bvhar::LdltInits ldlt_inits(init_spec);
				sur_objs[i].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 2: {
				bvhar::SsvsParams ssvs_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				bvhar::SsvsInits ssvs_inits(init_spec);
				sur_objs[i].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 3: {
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				bvhar::HsInits hs_inits(init_spec);
				sur_objs[i].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
			case 4: {
				bvhar::HierminnParams minn_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				bvhar::HierminnInits minn_inits(init_spec);
				sur_objs[i].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i])));
				break;
			}
		}
	}
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; ++window) {
		for (int i = 0; i < num_iter; ++i) {
			sur_objs[window]->doPosteriorDraws();
		}
		bvhar::LdltRecords reg_record = sur_objs[window]->returnLdltRecords(num_burn, thin);
		spillover[window].reset(new bvhar::RegVharSpillover(reg_record, step, month, har_trans));
		spillover[window]->computeSpillover();
		to_sp.row(window) = spillover[window]->returnTo();
		from_sp.row(window) = spillover[window]->returnFrom();
		tot[window] = spillover[window]->returnTot();
		sur_objs[window].reset();
		spillover[window].reset();
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}
