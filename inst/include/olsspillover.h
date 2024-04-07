#ifndef OLSSPILLOVER_H
#define OLSSPILLOVER_H

#include "ols.h"
#include "bvharstructural.h"

namespace bvhar {

class OlsSpillover {
public:
	OlsSpillover(const StructuralFit& fit)
	: step(fit._lag_max), dim(fit.dim),
		coef(fit._coef), cov(fit._cov), vma_mat(fit._vma),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)) {}
	virtual ~OlsSpillover() = default;
	// void computeFevd() {
	// 	fevd = compute_vma_fevd(vma_mat, cov, true);
	// }
	void computeSpillover() {
		fevd = compute_vma_fevd(vma_mat, cov, true);
		spillover = compute_sp_index(fevd);
	}
	Eigen::MatrixXd returnFevd() {
		return fevd;
	}
	Eigen::MatrixXd returnSpillover() {
		return spillover;
	}
	Eigen::VectorXd returnTo() {
		return compute_to(spillover);
	}
	Eigen::VectorXd returnFrom() {
		return compute_from(spillover);
	}
	double returnTot() {
		return compute_tot(spillover);
	}
	Eigen::MatrixXd returnNet() {
		return compute_net(spillover);
	}
protected:
	int step;
	int dim;
	Eigen::MatrixXd coef;
	Eigen::MatrixXd cov;
	Eigen::MatrixXd vma_mat;
	Eigen::MatrixXd fevd;
	Eigen::MatrixXd spillover;
};

}; // namespace bvhar

#endif // OLSSPILLOVER_H