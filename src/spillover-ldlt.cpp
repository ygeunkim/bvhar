#include "bvharomp.h"
#include <regspillover.h>
// #include <bvharspillover.h>
#include <algorithm>

// [[Rcpp::export]]
Rcpp::List compute_varldlt_spillover(int lag, int step, Rcpp::List fit_record, bool sparse) {
	// bvhar::LdltRecords reg_record(alpha_record, a_record, d_record);
	std::unique_ptr<bvhar::LdltRecords> reg_record;
	std::string coef_name = sparse ? "alpha_sparse_record" : "alpha_record";
	std::string a_name = sparse ? "a_sparse_record" : "a_record";
	std::string c_name = sparse ? "c_sparse_record" : "c_record";
	bvhar::initialize_record(reg_record, 0, fit_record, false, coef_name, a_name, c_name);
	auto spillover = std::make_unique<bvhar::RegSpillover>(*reg_record, step, lag);
	return spillover->returnSpilloverDensity();
}

// [[Rcpp::export]]
Rcpp::List compute_vharldlt_spillover(int week, int month, int step, Rcpp::List fit_record, bool sparse) {
	std::unique_ptr<bvhar::LdltRecords> reg_record;
	std::string coef_name = sparse ? "phi_sparse_record" : "phi_record";
	std::string a_name = sparse ? "a_sparse_record" : "a_record";
	std::string c_name = sparse ? "c_sparse_record" : "c_record";
	bvhar::initialize_record(reg_record, 0, fit_record, false, coef_name, a_name, c_name);
	int dim = reg_record->getDim();
	// int dim = d_record.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, false);
	auto spillover = std::make_unique<bvhar::RegVharSpillover>(*reg_record, step, month, har_trans);
	// bvhar::LdltRecords reg_record(phi_record, a_record, d_record);
	// auto spillover = std::make_unique<bvhar::RegVharSpillover>(reg_record, step, month, har_trans);
	return spillover->returnSpilloverDensity();
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvarldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			int lag, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			int prior_type, bool ggl, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
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
	// Eigen::MatrixXd tot(num_horizon, num_chains);
	// std::vector<Eigen::MatrixXd> to_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	// std::vector<Eigen::MatrixXd> from_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::vector<std::vector<Eigen::VectorXd>> tot(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> to_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> from_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> net_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	// std::vector<Eigen::MatrixXd> net(num_horizon, Eigen::MatrixXd(num_horizon, y.cols()));
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, lag, lag + 1);
		Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, lag, include_mean);
		if (ggl) {
			sur_objs[i] = bvhar::initialize_mcmc<bvhar::McmcReg, true>(
				num_chains, num_iter, roll_x0, roll_y0,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(i)
			);
		} else {
			sur_objs[i] = bvhar::initialize_mcmc<bvhar::McmcReg, false>(
				num_chains, num_iter, roll_x0, roll_y0,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(i)
			);
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
			// to_sp[0].row(window) = spillover[window][0]->returnTo();
			// from_sp[0].row(window) = spillover[window][0]->returnFrom();
			// tot(window, 0) = spillover[window][0]->returnTot();
			to_sp[window][0] = spillover[window][0]->returnTo();
			from_sp[window][0] = spillover[window][0]->returnFrom();
			tot[window][0] = spillover[window][0]->returnTot();
			net_sp[window][0] = to_sp[window][0] - from_sp[window][0];
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
				// to_sp[chain].row(window) = spillover[window][chain]->returnTo();
				// from_sp[chain].row(window) = spillover[window][chain]->returnFrom();
				// tot(window, chain) = spillover[window][chain]->returnTot();
				to_sp[window][chain] = spillover[window][chain]->returnTo();
				from_sp[window][chain] = spillover[window][chain]->returnFrom();
				tot[window][chain] = spillover[window][chain]->returnTot();
				net_sp[window][chain] = to_sp[window][chain] - from_sp[window][chain];
				spillover[window][chain].reset();
			}
		}
	}
	// std::vector<Eigen::MatrixXd> net(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	// std::transform(
	// 	to_sp.begin(), to_sp.end(), from_sp.begin(), net.begin(),
	// 	[](const Eigen::MatrixXd& to, const Eigen::MatrixXd& from) {
	// 		return to - from;
	// 	});
	return Rcpp::List::create(
		Rcpp::Named("to") = Rcpp::wrap(to_sp),
		Rcpp::Named("from") = Rcpp::wrap(from_sp),
		// Rcpp::Named("tot") = tot,
		// Rcpp::Named("net") = Rcpp::wrap(net)
		Rcpp::Named("tot") = Rcpp::wrap(tot),
		Rcpp::Named("net") = Rcpp::wrap(net_sp)
	);
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvharldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			 int week, int month, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			 int prior_type, bool ggl, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
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
	std::vector<std::vector<std::unique_ptr<bvhar::RegVharSpillover>>> spillover(num_horizon);
	for (auto &reg_spillover : spillover) {
		reg_spillover.resize(num_chains);
		for (auto &ptr : reg_spillover) {
			ptr = nullptr;
		}
	}
	// Eigen::MatrixXd tot(num_horizon, num_chains);
	// std::vector<Eigen::MatrixXd> to_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	// std::vector<Eigen::MatrixXd> from_sp(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	std::vector<std::vector<Eigen::VectorXd>> tot(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> to_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> from_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	std::vector<std::vector<Eigen::VectorXd>> net_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains));
	Eigen::MatrixXd har_trans = bvhar::build_vhar(y.cols(), week, month, include_mean);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, month, month + 1);
		Eigen::MatrixXd roll_x1 = bvhar::build_x0(roll_mat, month, include_mean) * har_trans.transpose();
		if (ggl) {
			sur_objs[i] = bvhar::initialize_mcmc<bvhar::McmcReg, true>(
				num_chains, num_iter, roll_x1, roll_y0,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(i)
			);
		} else {
			sur_objs[i] = bvhar::initialize_mcmc<bvhar::McmcReg, false>(
				num_chains, num_iter, roll_x1, roll_y0,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(i)
			);
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		for (int i = 0; i < num_iter; ++i) {
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
			// to_sp[0].row(window) = spillover[window][0]->returnTo();
			// from_sp[0].row(window) = spillover[window][0]->returnFrom();
			// tot(window, 0) = spillover[window][0]->returnTot();
			to_sp[window][0] = spillover[window][0]->returnTo();
			from_sp[window][0] = spillover[window][0]->returnFrom();
			tot[window][0] = spillover[window][0]->returnTot();
			net_sp[window][0] = to_sp[window][0] - from_sp[window][0];
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
				// to_sp[chain].row(window) = spillover[window][chain]->returnTo();
				// from_sp[chain].row(window) = spillover[window][chain]->returnFrom();
				// tot(window, chain) = spillover[window][chain]->returnTot();
				to_sp[window][chain] = spillover[window][chain]->returnTo();
				from_sp[window][chain] = spillover[window][chain]->returnFrom();
				tot[window][chain] = spillover[window][chain]->returnTot();
				net_sp[window][chain] = to_sp[window][chain] - from_sp[window][chain];
				spillover[window][chain].reset();
			}
		}
	}
	// std::vector<Eigen::MatrixXd> net(num_chains, Eigen::MatrixXd(num_horizon, y.cols()));
	// std::transform(
	// 	to_sp.begin(), to_sp.end(), from_sp.begin(), net.begin(),
	// 	[](const Eigen::MatrixXd& to, const Eigen::MatrixXd& from) {
	// 		return to - from;
	// 	});
	return Rcpp::List::create(
		Rcpp::Named("to") = Rcpp::wrap(to_sp),
		Rcpp::Named("from") = Rcpp::wrap(from_sp),
		// Rcpp::Named("tot") = tot,
		// Rcpp::Named("net") = Rcpp::wrap(net)
		Rcpp::Named("tot") = Rcpp::wrap(tot),
		Rcpp::Named("net") = Rcpp::wrap(net_sp)
	);
}
