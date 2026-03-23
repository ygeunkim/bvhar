#ifndef BVHAR_BAYES_MNIW_IRF_H
#define BVHAR_BAYES_MNIW_IRF_H

#include "../irf.h"
#include "./minnesota.h"
#include "../../math/random.h"

namespace baecon {
namespace bvhar {

class MinnIrf;
class MinnVharIrf;
class MinnIrfRun;

class MinnIrf : public McmcIrf {
public:
	MinnIrf(
		const MinnFit& fit, int lag_max, int num_iter, int num_burn, int thin, int ord, unsigned int seed,
		bool orthogonal = true
	)
	: McmcIrf(lag_max, ord, static_cast<int>((num_iter - num_burn + thin - 1) / thin), orthogonal),
		coef_mat(fit._coef), dim(coef_mat.cols()),
		cov(fit._prec.selfadjointView<Eigen::Lower>().llt().solve(Eigen::MatrixXd::Identity(dim, dim))),
		iw_scale(fit._iw_scale), vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)), iw_shape(fit._iw_shape),
		num_iter(num_iter), num_burn(num_burn), thin(thin),
		record_warm(num_burn, std::vector<Eigen::MatrixXd>(2)),
		record(num_iter - num_burn, std::vector<Eigen::MatrixXd>(2)),
		rng(seed) {
		vma_record = Eigen::MatrixXd::Zero(dim * step, num_sim * dim);
		for (int i = 0; i < num_burn; ++i) {
			record_warm[i] = sim_mn_iw(coef_mat, cov, iw_scale, iw_shape, false, rng);
		}
		for (int i = 0; i < num_iter - num_burn; ++i) {
			record[i] = sim_mn_iw(coef_mat, cov, iw_scale, iw_shape, false, rng);
		}
		if (thin > 1) {
			int id = 0;
			for (size_t thin_id = thin; thin_id < record.size(); thin_id += thin) {
				std::swap(record[id], record[thin_id]); // Move thin_id-th to the first num_iter - num_burn elements
				id++;
			}
			record.erase(record.begin() + id, record.end());
		}
	}

	virtual ~MinnIrf() = default;

protected:
	Eigen::MatrixXd coef_mat;
	int dim;
	Eigen::MatrixXd cov, iw_scale, vma_mat;
	double iw_shape;
	int num_iter, num_burn, thin;
	// MinnRecords mn_record;
	std::vector<std::vector<Eigen::MatrixXd>> record_warm;
	std::vector<std::vector<Eigen::MatrixXd>> record;
	BVHAR_BHRNG rng;

	void updateParams(const int i) override {
		coef_mat = record[i][0];
		cov = record[i][1];
	}

	void updateMovingAverage(const int i) override {
		if (orthogonal) {
			vma_mat = convert_vma_ortho(coef_mat, cov, lag, step - 1);
		} else {
			vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
		}
		vma_record.middleCols(i * dim, dim) = vma_mat;
	}
};

class MinnVharIrf : public MinnIrf {
public:
	MinnVharIrf(
		const MinnFit& fit, int lag_max, int num_iter, int num_burn, int thin,
		int month, unsigned int seed, const Eigen::MatrixXd har_trans,
		bool orthogonal = true
	)
	: MinnIrf(fit, lag_max, num_iter, num_burn, thin, month, seed, orthogonal),
		har_trans(har_trans) {}

	MinnVharIrf(
		const MinnFit& fit, int lag_max, int num_iter, int num_burn, int thin,
		int week, int month, unsigned int seed,
		bool orthogonal = true
	)
	: MinnIrf(fit, lag_max, num_iter, num_burn, thin, month, seed, orthogonal),
		har_trans(build_vhar(dim, week, month, false)) {}

	virtual ~MinnVharIrf() = default;

protected:
	void updateMovingAverage(const int i) override {
		if (orthogonal) {
			vma_mat = convert_vhar_vma_ortho(coef_mat, cov, har_trans, step - 1, lag);
		} else {
			vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
		}
		vma_record.middleCols(i * dim, dim) = vma_mat;
	}

private:
	Eigen::MatrixXd har_trans;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MNIW_IRF_H