#include "bvharomp.h"
#include "regspillover.h"
#include <algorithm>

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
Rcpp::List dynamic_bvarldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			int lag, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			int prior_type, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rf_error("Window size is too large.");
	}
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> sur_objs(num_horizon);
	for (auto &reg_chain : sur_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegSpillover>>> spillover(num_horizon);
	for (auto &reg_spillover : spillover) {
		reg_spillover.resize(num_chains);
		for (auto &ptr : reg_spillover) {
			ptr = nullptr;
		}
	}
	Eigen::MatrixXd tot(num_horizon, num_chains);
	std::vector<Eigen::MatrixXd> to_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::vector<Eigen::MatrixXd> from_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, lag, lag + 1);
		Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, lag, include_mean);
		switch (prior_type) {
			case 1: {
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, roll_x0, roll_y0,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits ldlt_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 2: {
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits<bvhar::LdltInits> ssvs_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 3: {
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits<bvhar::LdltInits> hs_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 4: {
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits<bvhar::LdltInits> minn_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 5: {
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits<bvhar::LdltInits> ng_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 6: {
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, roll_x0, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits<bvhar::LdltInits> dl_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		for (int i = 0; i < num_iter; i++) {
			sur_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::LdltRecords reg_record = sur_objs[window][chain]->returnLdltRecords(num_burn, thin, sparse);
		spillover[window][chain].reset(new bvhar::RegSpillover(reg_record, step, lag));
		sur_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; ++window) {
			run_gibbs(window, 0);
			spillover[window][0]->computeSpillover();
			to_sp[0].row(window) = spillover[window][0]->returnTo();
			from_sp[0].row(window) = spillover[window][0]->returnFrom();
			tot(window, 0) = spillover[window][0]->returnTot();
			spillover[window][0].reset();
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				run_gibbs(window, chain);
				spillover[window][chain]->computeSpillover();
				to_sp[chain].row(window) = spillover[window][chain]->returnTo();
				from_sp[chain].row(window) = spillover[window][chain]->returnFrom();
				tot(window, chain) = spillover[window][chain]->returnTot();
				spillover[window][chain].reset();
			}
		}
	}
	std::vector<Eigen::MatrixXd> net(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::transform(
		to_sp.begin(), to_sp.end(), from_sp.begin(), net.begin(),
		[](const Eigen::MatrixXd& to, const Eigen::MatrixXd& from) {
			return to - from;
		});
	return Rcpp::List::create(
		Rcpp::Named("to") = Rcpp::wrap(to_sp),
		Rcpp::Named("from") = Rcpp::wrap(from_sp),
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = Rcpp::wrap(net)
	);
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvharldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			 int week, int month, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			 int prior_type, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			 bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rf_error("Window size is too large.");
	}
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> sur_objs(num_horizon);
	for (auto &reg_chain : sur_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegSpillover>>> spillover(num_horizon);
	for (auto &reg_spillover : spillover) {
		reg_spillover.resize(num_chains);
		for (auto &ptr : reg_spillover) {
			ptr = nullptr;
		}
	}
	Eigen::MatrixXd tot(num_horizon, num_chains);
	std::vector<Eigen::MatrixXd> to_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::vector<Eigen::MatrixXd> from_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	Eigen::MatrixXd har_trans = bvhar::build_vhar(y.cols(), week, month, include_mean);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, month, month + 1);
		Eigen::MatrixXd roll_x1 = bvhar::build_x0(roll_mat, month, include_mean) * har_trans.transpose();
		switch (prior_type) {
			case 1: {
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, roll_x1, roll_y0,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits ldlt_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 2: {
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits<bvhar::LdltInits> ssvs_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 3: {
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits<bvhar::LdltInits> hs_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 4: {
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits<bvhar::LdltInits> minn_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 5: {
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits<bvhar::LdltInits> ng_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
			case 6: {
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, roll_x1, roll_y0,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; ++chain) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits<bvhar::LdltInits> dl_inits(init_spec);
					sur_objs[i][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(i, chain))));
				}
				break;
			}
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		for (int i = 0; i < num_iter; i++) {
			sur_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::LdltRecords reg_record = sur_objs[window][chain]->returnLdltRecords(num_burn, thin, sparse);
		spillover[window][chain].reset(new bvhar::RegVharSpillover(reg_record, step, month, har_trans));
		sur_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; ++window) {
			run_gibbs(window, 0);
			spillover[window][0]->computeSpillover();
			to_sp[0].row(window) = spillover[window][0]->returnTo();
			from_sp[0].row(window) = spillover[window][0]->returnFrom();
			tot(window, 0) = spillover[window][0]->returnTot();
			spillover[window][0].reset();
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				run_gibbs(window, chain);
				spillover[window][chain]->computeSpillover();
				to_sp[chain].row(window) = spillover[window][chain]->returnTo();
				from_sp[chain].row(window) = spillover[window][chain]->returnFrom();
				tot(window, chain) = spillover[window][chain]->returnTot();
				spillover[window][chain].reset();
			}
		}
	}
	std::vector<Eigen::MatrixXd> net(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::transform(
		to_sp.begin(), to_sp.end(), from_sp.begin(), net.begin(),
		[](const Eigen::MatrixXd& to, const Eigen::MatrixXd& from) {
			return to - from;
		});
	return Rcpp::List::create(
		Rcpp::Named("to") = Rcpp::wrap(to_sp),
		Rcpp::Named("from") = Rcpp::wrap(from_sp),
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = Rcpp::wrap(net)
	);
}
