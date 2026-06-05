#ifndef BVHAR_OLS_IRF_H
#define BVHAR_OLS_IRF_H

#include "../core/irf.h"
#include "./ols.h"

namespace baecon {
namespace bvhar {

// class OlsIrf;
class OlsVarIrf;
class OlsVharIrf;
class OlsIrfRun;

// class OlsIrf : public ImpulseResponse {
// public:
// 	OlsIrf(const StructuralFit& fit, bool orthogonal = true)
// 	: ImpulseResponse(fit._lag_max, fit._ord, orthogonal),
// 		dim(fit.dim), coef_mat(fit._coef), cov(fit._cov), vma_mat(fit._vma) {}
	
// 	OlsIrf(const StructuralFit& fit, int lag_max, bool orthogonal = true)
// 	: ImpulseResponse(lag_max, fit._ord, orthogonal),
// 		dim(fit.dim), coef_mat(fit._coef), cov(fit._cov),
// 		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)) {}
	
// 	virtual ~OlsIrf() = default;
	
// 	void computeIrf() override {
// 		computeVma();
// 	}

// 	Eigen::MatrixXd returnIrfDensity() {
// 		// computeIrf();
// 		computeVma();
// 		return vma_mat;
// 	}

// protected:
// 	int dim;
// 	Eigen::MatrixXd coef_mat, cov, vma_mat;

// 	virtual void computeVma() = 0;
// };

class OlsVarIrf : public ImpulseResponse {
public:
	OlsVarIrf(const StructuralFit& fit, bool orthogonal = true)
	: ImpulseResponse(fit._lag_max, fit._ord, orthogonal),
		dim(fit.dim), coef_mat(fit._coef), cov(fit._cov), vma_mat(fit._vma) {}
	
	OlsVarIrf(const StructuralFit& fit, int lag_max, bool orthogonal = true)
	: ImpulseResponse(lag_max, fit._ord, orthogonal),
		dim(fit.dim), coef_mat(fit._coef), cov(fit._cov),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)) {}
	
	virtual ~OlsVarIrf() = default;

	void computeIrf() override {
		computeVma();
	}

	Eigen::MatrixXd returnIrfDensity() {
		// computeIrf();
		computeVma();
		return vma_mat;
	}

protected:
	int dim;
	Eigen::MatrixXd coef_mat, cov, vma_mat;

	virtual void computeVma() {
		if (orthogonal) {
			vma_mat = convert_vma_ortho(coef_mat, cov, lag, step - 1);
		} else {
			vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
		}
	}
};

class OlsVharIrf : public OlsVarIrf {
public:
	OlsVharIrf(const StructuralFit& fit, int week, bool orthogonal = true)
	: OlsVarIrf(fit, orthogonal),
		har_trans(build_vhar(dim, week, lag, false)) {}
	
	OlsVharIrf(const StructuralFit& fit, int week, int lag_max, bool orthogonal = true)
	: OlsVarIrf(fit, lag_max, orthogonal),
		har_trans(build_vhar(dim, week, lag, false)) {}
	
	virtual ~OlsVharIrf() = default;

protected:
	void computeVma() override {
		if (orthogonal) {
			vma_mat = convert_vhar_vma_ortho(coef_mat, cov, har_trans, step - 1, lag);
		} else {
			vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
		}
	}

private:
	Eigen::MatrixXd har_trans;
};

inline std::unique_ptr<OlsVarIrf> initialize_olsirf(
	const Eigen::MatrixXd& coef_mat, int lag, const Eigen::MatrixXd& cov_mat, int step,
	bool orthogonal = true,
	BVHAR_OPTIONAL<int> week = BVHAR_NULLOPT
) {
	StructuralFit fit(coef_mat, lag, cov_mat);
	std::unique_ptr<OlsVarIrf> irf_ptr;
	if (week) {
		irf_ptr = std::make_unique<OlsVharIrf>(fit, *week, step, orthogonal);
	} else {
		irf_ptr = std::make_unique<OlsVarIrf>(fit, step, orthogonal);
	}
	return irf_ptr;
}

class OlsIrfRun : public IrfRun {
public:
	OlsIrfRun(int lag, int step, const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& cov_mat, bool orthogonal = true)
	: irf_ptr(initialize_olsirf(coef_mat, lag, cov_mat, step, orthogonal)) {}

	OlsIrfRun(int week, int month, int step, const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& cov_mat, bool orthogonal = true)
	: irf_ptr(initialize_olsirf(coef_mat, month, cov_mat, step, orthogonal, week)) {}
	
	virtual ~OlsIrfRun() = default;

	void computeIrf() override {
		irf_ptr->computeIrf();
	}
	
	// std::vector<Eigen::MatrixXd> returnIrf() {
	// 	computeIrf();
	// 	return density_irf;
	// }
	Eigen::MatrixXd returnIrf() {
		return irf_ptr->returnIrfDensity();
	}

protected:
	// int num_chains, nthreads;
	// std::vector<std::unique_ptr<McmcIrf>> irf_ptr;
	std::unique_ptr<OlsVarIrf> irf_ptr;
	// std::vector<Eigen::MatrixXd> density_irf;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_OLS_IRF_H