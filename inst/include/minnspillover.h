#ifndef MINNSPILLOVER_H
#define MINNSPILLOVER_H

#include "minnesota.h"
#include "bvharsim.h"
#include "bvharstructural.h"

namespace bvhar {

class MinnSpillover {
public:
	MinnSpillover(const MinnFit& fit, int lag_max, int num_iter, int num_burn, int ord, unsigned int seed)
	: coef(fit._coef), cov(fit._prec.inverse()), iw_scale(fit._iw_scale), iw_shape(fit._iw_shape),
		step(lag_max), dim(coef.cols()),
		num_iter(num_iter), num_burn(num_burn), lag(ord),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)),
		record_warm(num_burn, std::vector<Eigen::MatrixXd>(2)),
		record(num_iter - num_burn, std::vector<Eigen::MatrixXd>(2)),
		rng(seed) {}
	virtual ~MinnSpillover() = default;
	void updateMniw() {
		for (int i = 0; i < num_burn; ++i) {
			record_warm[i] = sim_mn_iw(coef, cov, iw_scale, iw_shape, rng);
		}
		for (int i = 0; i < num_iter - num_burn; ++i) {
			record[i] = sim_mn_iw(coef, cov, iw_scale, iw_shape, rng);
		}
	}
	virtual void computeSpillover() {
	// virtual void computeSpillover(int nthreads) {
		// MNIW before this method
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads) private(vma_mat)
	// #endif
		for (int j = 0; j < num_iter - num_burn; ++j) {
			vma_mat = convert_var_to_vma(record[j][0], lag, step - 1);
			fevd += compute_vma_fevd(vma_mat, record[j][1], true);
		}
		fevd /= (num_iter - num_burn);
		spillover = compute_sp_index(fevd);
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
	Eigen::MatrixXd coef;
	Eigen::MatrixXd cov;
	Eigen::MatrixXd iw_scale;
	double iw_shape;
	int step;
	int dim;
	int num_iter;
	int num_burn;
	int lag; // p of VAR or month of VHAR
	Eigen::MatrixXd vma_mat;
	Eigen::MatrixXd fevd;
	Eigen::MatrixXd spillover;
	std::vector<std::vector<Eigen::MatrixXd>> record_warm;
	std::vector<std::vector<Eigen::MatrixXd>> record;
	boost::random::mt19937 rng;
};

class BvharSpillover : public MinnSpillover {
public:
	BvharSpillover(const MinnFit& fit, int lag_max, int num_iter, int num_burn, int ord, const Eigen::MatrixXd& har_trans, unsigned int seed)
	: MinnSpillover(fit, lag_max, num_iter, num_burn, ord, seed), har_trans(har_trans) {}
	virtual ~BvharSpillover() = default;
	void computeSpillover() override {
	// void computeSpillover(int nthreads) override {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads) private(vma_mat)
	// #endif
		for (int j = 0; j < num_iter - num_burn; ++j) {
			vma_mat = convert_vhar_to_vma(record[j][0], har_trans, step - 1, lag);
			fevd += compute_vma_fevd(vma_mat, record[j][1], true);
		}
		fevd /= (num_iter - num_burn);
		spillover = compute_sp_index(fevd);
	}
private:
	Eigen::MatrixXd har_trans;
};


}; // namespace bvhar

#endif // MINNSPILLOVER_H