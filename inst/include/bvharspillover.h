#ifndef BVHARSPILLOVER_H
#define BVHARSPILLOVER_H

#include "bvharmcmc.h"
#include "bvharstructural.h"

namespace bvhar {

class McmcSpillover;
template <typename RecordType> class McmcVarSpillover;
template <typename RecordType> class McmcVharSpillover;
template <typename RecordType> class McmcSpilloverRun;
class DynamicLdltSpillover;
class DynamicSvSpillover;

/**
 * @brief Spillover class for `McmcTriangular`
 * 
 */
class McmcSpillover {
public:
	McmcSpillover(const RegRecords& records, int lag_max, int ord, int dim, int id = -1)
	: step(lag_max), time_id(id), lag(ord), dim(dim),
		num_coef(records.coef_record.cols()), num_sim(records.coef_record.rows()),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		sqrt_sig(Eigen::MatrixXd::Zero(dim, dim)),
		cov(Eigen::MatrixXd::Zero(dim, dim)),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)),
		fevd(Eigen::MatrixXd::Zero(dim * step, num_sim * dim)),
		spillover(Eigen::MatrixXd::Zero(dim, num_sim * dim)),
		to_spillover(Eigen::VectorXd::Zero(num_sim * dim)),
		from_spillover(Eigen::VectorXd::Zero(num_sim * dim)),
		tot_spillover(Eigen::VectorXd::Zero(num_sim)),
		net_spillover(Eigen::MatrixXd::Zero(dim, num_sim * dim)) {}
	virtual ~McmcSpillover() = default;
	
	/**
	 * @brief Generate spillover density
	 * 
	 */
	void computeSpillover() {
		for (int i = 0; i < num_sim; ++i) {
			reg_record->updateDiag(i, time_id, sv_update);
			sqrt_sig = build_inv_lower(
				dim,
				reg_record->contem_coef_record.row(i)
			).triangularView<Eigen::UnitLower>().solve(sv_update.asDiagonal().toDenseMatrix());
			cov = sqrt_sig * sqrt_sig.transpose();
			coef_mat = unvectorize(reg_record->coef_record.row(i).transpose(), dim);
			computeVma();
			fevd.middleCols(i * dim, dim) = compute_vma_fevd(vma_mat, cov, true);
			spillover.middleCols(i * dim, dim) = compute_sp_index(fevd.middleCols(i * dim, dim));
			to_spillover.segment(i * dim, dim) = compute_to(spillover.middleCols(i * dim, dim));
			from_spillover.segment(i * dim, dim) = compute_from(spillover.middleCols(i * dim, dim));
			tot_spillover[i] = compute_tot(spillover.middleCols(i * dim, dim));
			net_spillover.middleCols(i * dim, dim) = compute_net(spillover.middleCols(i * dim, dim));
		}
	}

	/**
	 * @brief Return spillover density
	 * 
	 * @return LIST Every spillover-related density
	 */
	LIST returnSpilloverDensity() {
		computeSpillover();
		LIST res = CREATE_LIST(
			NAMED("connect") = spillover,
			NAMED("to") = to_spillover,
			NAMED("from") = from_spillover,
			NAMED("tot") = tot_spillover,
			NAMED("net") = to_spillover - from_spillover,
			NAMED("net_pairwise") = net_spillover
		);
		return res;
	}

	/**
	 * @brief Return FEVD
	 * 
	 * @return Eigen::MatrixXd Forecast error variance decomposition density
	 */
	Eigen::MatrixXd returnFevd() {
		return fevd;
	}

	/**
	 * @brief Return spillover
	 * 
	 * @return Eigen::MatrixXd Spillover density 
	 */
	Eigen::MatrixXd returnSpillover() {
		return spillover;
	}

	/**
	 * @brief Return to-spillover
	 * 
	 * @return Eigen::VectorXd to-spillover density
	 */
	Eigen::VectorXd returnTo() {
		return to_spillover;
	}

	/**
	 * @brief Return from-spillover
	 * 
	 * @return Eigen::VectorXd from-spillover density
	 */
	Eigen::VectorXd returnFrom() {
		return from_spillover;
	}

	/**
	 * @brief Return total spillover
	 * 
	 * @return Eigen::VectorXd total spillover density
	 */
	Eigen::VectorXd returnTot() {
		return tot_spillover;
	}

	/**
	 * @brief Return net spillover
	 * 
	 * @return Eigen::MatrixXd Net spillover density
	 */
	Eigen::MatrixXd returnNet() {
		return net_spillover;
	}

protected:
	int step;
	int time_id;
	int lag; // p of VAR or month of VHAR
	int dim;
	int num_coef;
	int num_sim;
	std::unique_ptr<RegRecords> reg_record;
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::VectorXd sv_update; // D_t^(1 / 2)
	Eigen::MatrixXd sqrt_sig; // L^(-1) D_t(1 / 2)
	Eigen::MatrixXd cov; // Sigma_t
	Eigen::MatrixXd vma_mat;
	Eigen::MatrixXd fevd; // rbind(step), cbind(sims)
	Eigen::MatrixXd spillover; // rbind(step), cbind(sims)
	Eigen::VectorXd to_spillover;
	Eigen::VectorXd from_spillover;
	Eigen::VectorXd tot_spillover;
	Eigen::MatrixXd net_spillover;

	/**
	 * @brief VMA representation
	 * 
	 */
	virtual void computeVma() = 0;
};

/**
 * @brief Spillover class for VAR with `McmcTriangular`
 * 
 * @tparam RecordType `LdltRecords` or `SvRecords`
 */
template <typename RecordType = LdltRecords>
class McmcVarSpillover : public McmcSpillover {
public:
	McmcVarSpillover(RecordType& records, int lag_max, int ord, int id = -1)
	: McmcSpillover(records, lag_max, ord, records.getDim(), id) {
		reg_record = std::make_unique<RecordType>(records);
	}
	virtual ~McmcVarSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
	}
};

/**
 * @brief Spillover class for VHAR with `McmcTriangular`
 * 
 * @tparam RecordType `LdltRecords` or `SvRecords`
 */
template <typename RecordType = LdltRecords>
class McmcVharSpillover : public McmcSpillover {
public:
	McmcVharSpillover(RecordType& records, int lag_max, int month, const Eigen::MatrixXd& har_trans, int id = -1)
	: McmcSpillover(records, lag_max, month, records.getDim(), id), har_trans(har_trans) {
		reg_record = std::make_unique<RecordType>(records);
	}
	McmcVharSpillover(RecordType& records, int lag_max, int week, int month, int id = -1)
	: McmcSpillover(records, lag_max, month, records.getDim(), id),
		har_trans(build_vhar(records.getDim(), week, month, false)) {
		reg_record = std::make_unique<RecordType>(records);
	}
	virtual ~McmcVharSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	}

private:
	Eigen::MatrixXd har_trans; // without constant term
};

template <typename RecordType = LdltRecords>
inline std::unique_ptr<McmcSpillover> initialize_spillover(
	int chain_id, int lag, int step, LIST& fit_record, bool sparse, int id = -1,
	Optional<Eigen::MatrixXd> har_trans = NULLOPT, Optional<int> week = NULLOPT
) {
	std::unique_ptr<RecordType> reg_record;
	STRING coef_name = (har_trans || week) ? (sparse ? "phi_sparse_record" : "phi_record") : (sparse ? "alpha_sparse_record" : "alpha_record");
	STRING a_name = sparse ? "a_sparse_record" : "a_record";
	STRING c_name = sparse ? "c_sparse_record" : "c_record";
	initialize_record(reg_record, chain_id, fit_record, false, coef_name, a_name, c_name);
	std::unique_ptr<McmcSpillover> spillover_ptr;
	if (har_trans) {
		spillover_ptr = std::make_unique<McmcVharSpillover<RecordType>>(*reg_record, step, lag, *har_trans, id);
	} else if (week) {
		spillover_ptr = std::make_unique<McmcVharSpillover<RecordType>>(*reg_record, step, *week, lag, id);
	} else {
		spillover_ptr = std::make_unique<McmcVarSpillover<RecordType>>(*reg_record, step, lag, id);
	}
	return spillover_ptr;
}

template <typename RecordType = LdltRecords>
inline std::unique_ptr<McmcSpillover> initialize_spillover(
	int lag, int step, RecordType& reg_record, int id = -1,
	Optional<Eigen::MatrixXd> har_trans = NULLOPT, Optional<int> week = NULLOPT
) {
	std::unique_ptr<McmcSpillover> spillover_ptr;
	if (har_trans) {
		spillover_ptr = std::make_unique<McmcVharSpillover<RecordType>>(reg_record, step, lag, *har_trans, id);
	} else if (week) {
		spillover_ptr = std::make_unique<McmcVharSpillover<RecordType>>(reg_record, step, *week, lag, id);
	} else {
		spillover_ptr = std::make_unique<McmcVarSpillover<RecordType>>(reg_record, step, lag, id);
	}
	return spillover_ptr;
}

/**
 * @brief Spillover running class
 * 
 * @tparam RecordType `LdltRecords` or `SvRecords`
 */
template <typename RecordType = LdltRecords>
class McmcSpilloverRun {
public:
	McmcSpilloverRun(int lag, int step, LIST& fit_record, bool sparse)
	: spillover_ptr(initialize_spillover<RecordType>(0, lag, step, fit_record, sparse, -1)) {}
	McmcSpilloverRun(int week, int month, int step, LIST& fit_record, bool sparse)
	: spillover_ptr(initialize_spillover<RecordType>(0, month, step, fit_record, sparse, -1, NULLOPT, week)) {}
	virtual ~McmcSpilloverRun() = default;
	LIST returnSpillover() {
		return spillover_ptr->returnSpilloverDensity();
	}

private:
	std::unique_ptr<McmcSpillover> spillover_ptr;
};

/**
 * @brief Dynamic spillover class for `LdltRecords`
 * 
 */
class DynamicLdltSpillover {
public:
	DynamicLdltSpillover(
		const Eigen::MatrixXd& y, int window, int step, int lag, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type, bool ggl,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, const Eigen::MatrixXi& seed_chain, int nthreads
	)
	: num_horizon(y.rows() - window + 1), win_size(window), lag(lag), step(step),
		num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin),
		include_mean(include_mean), sparse(sparse),
		tot(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		to_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		from_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		net_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		model(num_horizon), spillover(num_horizon),
		har_trans(NULLOPT) {
		initialize(
			y, param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	DynamicLdltSpillover(
		const Eigen::MatrixXd& y, int window, int step, int week, int month, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type, bool ggl,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, const Eigen::MatrixXi& seed_chain, int nthreads
	)
	: num_horizon(y.rows() - window + 1), win_size(window), lag(month), step(step),
		num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		include_mean(include_mean), sparse(sparse),
		tot(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		to_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		from_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		net_sp(num_horizon, std::vector<Eigen::VectorXd>(num_chains)),
		model(num_horizon), spillover(num_horizon),
		har_trans(build_vhar(y.cols(), week, month, include_mean)) {
		if (num_horizon <= 0) {
			STOP("Window size is too large");
		}
		initialize(
			y, param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			grp_id, own_id, cross_id, grp_mat,
			seed_chain
		);
	}
	virtual ~DynamicLdltSpillover() = default;
	LIST returnSpillover() {
		fit();
		LIST res = CREATE_LIST(
			NAMED("to") = WRAP(to_sp),
			NAMED("from") = WRAP(from_sp),
			NAMED("tot") = WRAP(tot),
			NAMED("net") = WRAP(net_sp)
		);
		return res;
	}

protected:
	int num_horizon, win_size, lag, step, num_chains, num_iter, num_burn, thin, nthreads;
	bool include_mean, sparse;
	std::vector<std::vector<Eigen::VectorXd>> tot;
	std::vector<std::vector<Eigen::VectorXd>> to_sp;
	std::vector<std::vector<Eigen::VectorXd>> from_sp;
	std::vector<std::vector<Eigen::VectorXd>> net_sp;
	std::vector<std::vector<std::unique_ptr<McmcReg>>> model;
	std::vector<std::vector<std::unique_ptr<McmcSpillover>>> spillover;
	Optional<Eigen::MatrixXd> har_trans;

	/**
	 * @brief Initialize every member of `DynamicLdltSpillover`
	 * 
	 * @param y Response matrix
	 * @param param_reg `LIST` of CTA hyperparameters
	 * @param param_prior `LIST` of shrinkage prior hyperparameters
	 * @param param_intercept `LIST` of Normal prior hyperparameters for constant term
	 * @param param_init `LIST_OF_LIST` for initial values
	 * @param prior_type Shrinkage prior number
	 * @param ggl Group parameter?
	 * @param grp_id Minnesota group unique id
	 * @param own_id own-lag id
	 * @param cross_id cross-lag id
	 * @param grp_mat Minnesota group matrix
	 * @param seed_chain Random seed for each window and chain
	 */
	void initialize(
		const Eigen::MatrixXd& y, LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type, bool ggl,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) {
		for (auto &reg_chain : model) {
			reg_chain.resize(num_chains);
			for (auto &ptr : reg_chain) {
				ptr = nullptr;
			}
		}
		for (auto &reg_spillover : spillover) {
			reg_spillover.resize(num_chains);
			for (auto &ptr : reg_spillover) {
				ptr = nullptr;
			}
		}
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd roll_mat = y.middleRows(i, win_size);
			Eigen::MatrixXd roll_y0 = build_y0(roll_mat, lag, lag + 1);
			Eigen::MatrixXd roll_design = buildDesign(roll_mat, har_trans);
			if (ggl) {
				model[i] = initialize_mcmc<McmcReg, true>(
					num_chains, num_iter - num_burn, roll_design, roll_y0,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(i)
				);
			} else {
				model[i] = initialize_mcmc<McmcReg, false>(
					num_chains, num_iter - num_burn, roll_design, roll_y0,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(i)
				);
			}
		}
	}
	Eigen::MatrixXd buildDesign(Eigen::Ref<Eigen::MatrixXd> sample_mat, Optional<Eigen::MatrixXd> har_trans = NULLOPT) {
		if (har_trans) {
			return build_x0(sample_mat, lag, include_mean) * (*har_trans).transpose();
		}
		return build_x0(sample_mat, lag, include_mean);
	}
	void runGibbs(int window, int chain) {
		for (int i = 0; i < num_burn; ++i) {
			model[window][chain]->doWarmUp();
		}
		for (int i = num_burn; i < num_iter; ++i) {
			model[window][chain]->doPosteriorDraws();
		}
		LdltRecords reg_record = model[window][chain]->returnLdltRecords(0, thin, sparse);
		spillover[window][chain] = initialize_spillover<LdltRecords>(lag, step, reg_record, -1, har_trans);
		model[window][chain].reset();
	}
	void getSpillover(int window, int chain) {
		spillover[window][chain]->computeSpillover();
		to_sp[window][chain] = spillover[window][chain]->returnTo();
		from_sp[window][chain] = spillover[window][chain]->returnFrom();
		tot[window][chain] = spillover[window][chain]->returnTot();
		net_sp[window][chain] = to_sp[window][chain] - from_sp[window][chain];
		spillover[window][chain].reset();
	}
	void fit() {
		if (num_chains == 1) {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				runGibbs(window, 0);
				getSpillover(window, 0);
			}
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				for (int chain = 0; chain < num_chains; ++chain) {
					runGibbs(window, chain);
					getSpillover(window, chain);
				}
			}
		}
	}
};

/**
 * @brief Dynamic spillover class for `SvRecords`
 * 
 */
class DynamicSvSpillover {
public:
	DynamicSvSpillover(int lag, int step, int num_design, LIST& fit_record, bool include_mean, bool sparse, int nthreads)
	: num_horizon(num_design), lag(lag), step(step), nthreads(nthreads), sparse(sparse),
		tot(num_design), to_sp(num_design), from_sp(num_design), net_sp(num_design),
		spillover(num_horizon), har_trans(NULLOPT) {
		STRING coef_name = sparse ? "alpha_sparse_record" : "alpha_record";
		STRING a_name = sparse ? "a_sparse_record" : "a_record";
		STRING c_name = sparse ? "c_sparse_record" : "c_record";
		initialize_record(reg_record, 0, fit_record, include_mean, coef_name, a_name, c_name);
	}
	DynamicSvSpillover(int week, int month, int step, int num_design, LIST& fit_record, bool include_mean, bool sparse, int nthreads)
	: num_horizon(num_design), lag(month), step(step), nthreads(nthreads), sparse(sparse),
		tot(num_design), to_sp(num_design), from_sp(num_design), net_sp(num_design),
		spillover(num_horizon) {
		STRING coef_name = sparse ? "phi_sparse_record" : "phi_record";
		STRING a_name = sparse ? "a_sparse_record" : "a_record";
		STRING c_name = sparse ? "c_sparse_record" : "c_record";
		initialize_record(reg_record, 0, fit_record, include_mean, coef_name, a_name, c_name);
		har_trans = build_vhar(reg_record->getDim(), week, month, include_mean);
	}
	virtual ~DynamicSvSpillover() = default;
	LIST returnSpillover() {
		fit();
		LIST res = CREATE_LIST(
			NAMED("to") = WRAP(to_sp),
			NAMED("from") = WRAP(from_sp),
			NAMED("tot") = WRAP(tot),
			NAMED("net") = WRAP(net_sp)
		);
		return res;
	}

protected:
	int num_horizon, lag, step, nthreads;
	bool sparse;
	std::vector<Eigen::VectorXd> tot;
	std::vector<Eigen::VectorXd> to_sp;
	std::vector<Eigen::VectorXd> from_sp;
	std::vector<Eigen::VectorXd> net_sp;
	std::vector<std::unique_ptr<McmcSpillover>> spillover;
	std::unique_ptr<SvRecords> reg_record;
	Optional<Eigen::MatrixXd> har_trans;
	void fit() {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; ++window) {
			spillover[window] = initialize_spillover<SvRecords>(lag, step, *reg_record, window, har_trans);
			spillover[window]->computeSpillover();
			to_sp[window] = spillover[window]->returnTo();
			from_sp[window] = spillover[window]->returnFrom();
			tot[window] = spillover[window]->returnTot();
			net_sp[window] = to_sp[window] - from_sp[window];
			spillover[window].reset(); // free the memory by making nullptr
		}
	}
};

} // namespace bvhar

#endif // BVHARSPILLOVER_H