#ifndef BVHAR_OLS_SPILLOVER_H
#define BVHAR_OLS_SPILLOVER_H

#include "./ols.h"
#include "../math/structural.h"

namespace bvhar {

// class OlsSpillover;
class OlsVarSpillover;
class OlsVharSpillover;
class OlsSpilloverRun;
class OlsDynamicSpillover;

class OlsVarSpillover {
public:
	OlsVarSpillover(const StructuralFit& fit)
	: step(fit._lag_max), dim(fit.dim),
		coef(fit._coef), cov(fit._cov), vma_mat(fit._vma),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)), net_spillover(Eigen::MatrixXd::Zero(dim, dim)),
		to_spillover(Eigen::VectorXd::Zero(dim)),
		from_spillover(Eigen::VectorXd::Zero(dim)),
		tot_spillover(0) {}
	OlsVarSpillover(const StructuralFit& fit, int lag_max)
	: step(lag_max), dim(fit.dim), lag(fit._ord),
		coef(fit._coef), cov(fit._cov), vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)), net_spillover(Eigen::MatrixXd::Zero(dim, dim)),
		to_spillover(Eigen::VectorXd::Zero(dim)),
		from_spillover(Eigen::VectorXd::Zero(dim)),
		tot_spillover(0) {}
	virtual ~OlsVarSpillover() = default;
	// void computeFevd() {
	// 	fevd = compute_vma_fevd(vma_mat, cov, true);
	// }
	void computeSpillover() {
		computeVma();
		fevd = compute_vma_fevd(vma_mat, cov, true);
		spillover = compute_sp_index(fevd);
		to_spillover = compute_to(spillover);
		from_spillover = compute_from(spillover);
		tot_spillover = compute_tot(spillover);
		net_spillover = compute_net(spillover);
	}

	BVHAR_LIST returnSpilloverResult() {
		computeSpillover();
		BVHAR_LIST res = BVHAR_CREATE_LIST(
			BVHAR_NAMED("connect") = spillover,
			BVHAR_NAMED("to") = to_spillover,
			BVHAR_NAMED("from") = from_spillover,
			BVHAR_NAMED("tot") = tot_spillover,
			BVHAR_NAMED("net") = to_spillover - from_spillover,
			BVHAR_NAMED("net_pairwise") = net_spillover
		);
		return res;
	}

	Eigen::MatrixXd returnFevd() {
		return fevd;
	}
	Eigen::MatrixXd returnSpillover() {
		return spillover;
	}
	Eigen::VectorXd returnTo() {
		return to_spillover;
	}
	Eigen::VectorXd returnFrom() {
		return from_spillover;
	}
	double returnTot() {
		return tot_spillover;
	}
	Eigen::MatrixXd returnNet() {
		return net_spillover;
	}
protected:
	int step, dim, lag;
	Eigen::MatrixXd coef, cov;
	Eigen::MatrixXd vma_mat, fevd, spillover, net_spillover;
	Eigen::VectorXd to_spillover, from_spillover;
	double tot_spillover;

	virtual void computeVma() {
		vma_mat = convert_var_to_vma(coef, lag, step - 1);
	}
};

class OlsVharSpillover : public OlsVarSpillover {
public:
	OlsVharSpillover(const StructuralFit& fit, int week, int lag_max)
	: OlsVarSpillover(fit, lag_max),
		har_trans(build_vhar(dim, week, lag, false)) {}
	virtual ~OlsVharSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef, har_trans, step - 1, lag);
	}

private:
	Eigen::MatrixXd har_trans;
};

inline std::unique_ptr<OlsVarSpillover> initialize_olsspillover(
	const Eigen::MatrixXd& coef_mat, int lag, const Eigen::MatrixXd& cov_mat, int step,
	Optional<int> week = NULLOPT
) {
	StructuralFit fit(coef_mat, lag, cov_mat);
	std::unique_ptr<OlsVarSpillover> spillvoer_ptr;
	if (week) {
		spillvoer_ptr = std::make_unique<OlsVharSpillover>(fit, *week, step);
	} else {
		spillvoer_ptr = std::make_unique<OlsVarSpillover>(fit, step);
	}
	return spillvoer_ptr;
}

inline std::unique_ptr<OlsVarSpillover> initialize_olsspillover(const StructuralFit& fit, int step, Optional<int> week = NULLOPT) {
	std::unique_ptr<OlsVarSpillover> spillvoer_ptr;
	if (week) {
		spillvoer_ptr = std::make_unique<OlsVharSpillover>(fit, *week, step);
	} else {
		spillvoer_ptr = std::make_unique<OlsVarSpillover>(fit, step);
	}
	return spillvoer_ptr;
}

class OlsSpilloverRun {
public:
	OlsSpilloverRun(int lag, int step, const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& cov_mat)
	: spillover_ptr(initialize_olsspillover(coef_mat, lag, cov_mat, step)) {}
	OlsSpilloverRun(int week, int month, int step, const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& cov_mat)
	: spillover_ptr(initialize_olsspillover(coef_mat, month, cov_mat, step, week)) {}
	virtual ~OlsSpilloverRun() = default;
	BVHAR_LIST returnSpillover() {
		return spillover_ptr->returnSpilloverResult();
	}
	
private:
	std::unique_ptr<OlsVarSpillover> spillover_ptr;
};

class OlsDynamicSpillover {
public:
	OlsDynamicSpillover(
		const Eigen::MatrixXd& y, int window, int step, int lag, bool include_mean, int method, int nthreads,
		Optional<int> week = NULLOPT
	)
	: num_horizon(y.rows() - window + 1), win_size(window), lag(lag), step(step), nthreads(nthreads),
		ols_ptr(num_horizon), spillover(num_horizon),
		tot(Eigen::VectorXd::Zero(num_horizon)),
		to_sp(Eigen::MatrixXd::Zero(num_horizon, y.cols())),
		from_sp(Eigen::MatrixXd::Zero(num_horizon, y.cols())),
		har_week(week) {
		if (har_week) {
			for (int i = 0; i < num_horizon; ++i) {
				Eigen::MatrixXd roll_mat = y.middleRows(i, win_size);
				ols_ptr[i] = std::make_unique<OlsVhar>(roll_mat, *har_week, lag, include_mean, method);
			}
		} else {
			for (int i = 0; i < num_horizon; ++i) {
				Eigen::MatrixXd roll_mat = y.middleRows(i, win_size);
				ols_ptr[i] = std::make_unique<OlsVar>(roll_mat, lag, include_mean, method);
			}
		}
	}
	virtual ~OlsDynamicSpillover() = default;
	BVHAR_LIST returnSpillover() {
		fit();
		BVHAR_LIST res = BVHAR_CREATE_LIST(
			BVHAR_NAMED("to") = to_sp,
			BVHAR_NAMED("from") = from_sp,
			BVHAR_NAMED("tot") = tot,
			BVHAR_NAMED("net") = to_sp - from_sp
		);
		return res;
	}

private:
	int num_horizon, win_size, lag, step, nthreads;
	// bool include_mean;
	std::vector<std::unique_ptr<OlsInterface>> ols_ptr;
	std::vector<std::unique_ptr<OlsVarSpillover>> spillover;
	Eigen::VectorXd tot;
	Eigen::MatrixXd to_sp, from_sp;
	Optional<int> har_week;

	void getSpillover(int window) {
		StructuralFit ols_fit = ols_ptr[window]->returnStructuralFit();
		spillover[window] = initialize_olsspillover(ols_fit, step, har_week);
		spillover[window]->computeSpillover();
		to_sp.row(window) = spillover[window]->returnTo();
		from_sp.row(window) = spillover[window]->returnFrom();
		tot[window] = spillover[window]->returnTot();
		ols_ptr[window].reset();
		spillover[window].reset();
	}

	void fit() {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; ++window) {
			getSpillover(window);
		}
	}
};

}; // namespace bvhar

#endif // BVHAR_OLS_SPILLOVER_H