#ifndef BVHAR_BAYES_MNIW_SPILLOVER_H
#define BVHAR_BAYES_MNIW_SPILLOVER_H

#include "./minnesota.h"
#include "../../math/random.h"
#include "../../math/structural.h"

namespace bvhar {

class MinnSpillover;
class BvharSpillover;

class MinnSpillover {
public:
	MinnSpillover(const MinnFit& fit, int lag_max, int num_iter, int num_burn, int thin, int ord, unsigned int seed)
	: coef(fit._coef), prec(fit._prec), iw_scale(fit._iw_scale), iw_shape(fit._iw_shape),
	// MinnSpillover(const MinnRecords& records, int lag_max, int ord)
	// : mn_record(records),
		step(lag_max), dim(coef.cols()),
		// step(lag_max), dim(sqrt(mn_record.sig_record.cols())), num_sim(mn_record.coef_record.rows()),
		num_iter(num_iter), num_burn(num_burn), thin(thin), lag(ord),
		// coef_mat(Eigen::MatrixXd::Zero(mn_record.coef_record.cols() / dim, dim)),
		// cov_mat(Eigen::MatrixXd::Zero(dim, dim)),
		// lag(ord),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)),
		record_warm(num_burn, std::vector<Eigen::MatrixXd>(2)),
		record(num_iter - num_burn, std::vector<Eigen::MatrixXd>(2)),
		rng(seed) {}
	virtual ~MinnSpillover() = default;
	void updateMniw() {
		for (int i = 0; i < num_burn; ++i) {
			record_warm[i] = sim_mn_iw(coef, prec, iw_scale, iw_shape, true, rng);
		}
		for (int i = 0; i < num_iter - num_burn; ++i) {
			record[i] = sim_mn_iw(coef, prec, iw_scale, iw_shape, true, rng);
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
	// virtual void computeVma() {
	// 	vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
	// }
	virtual void computeSpillover() {
	// void computeSpillover() {
		for (size_t j = 0; j < record.size(); ++j) {
			vma_mat = convert_var_to_vma(record[j][0], lag, step - 1);
			fevd += compute_vma_fevd(vma_mat, record[j][1], true);
		}
		fevd /= static_cast<int>(record.size());
		// for (int j = 0; j < num_sim; ++j) {
		// 	coef_mat = unvectorize(mn_record.coef_record.row(j), dim);
		// 	cov_mat = unvectorize(mn_record.sig_record.row(j), dim);
		// 	computeVma();
		// 	fevd += compute_vma_fevd(vma_mat, cov_mat, true);
		// }
		// fevd /= num_sim;
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
	Eigen::MatrixXd coef;
	Eigen::MatrixXd prec;
	Eigen::MatrixXd iw_scale;
	double iw_shape;
	// MinnRecords mn_record;
	int step;
	int dim;
	// int num_sim;
	int num_iter;
	int num_burn;
	int thin;
	// Eigen::MatrixXd coef_mat;
	// Eigen::MatrixXd cov_mat;
	int lag; // p of VAR or month of VHAR
	Eigen::MatrixXd vma_mat;
	Eigen::MatrixXd fevd;
	Eigen::MatrixXd spillover;
	std::vector<std::vector<Eigen::MatrixXd>> record_warm;
	std::vector<std::vector<Eigen::MatrixXd>> record;
	BHRNG rng;
};

class BvharSpillover : public MinnSpillover {
public:
	BvharSpillover(const MinnFit& fit, int lag_max, int num_iter, int num_burn, int thin, int ord, const Eigen::MatrixXd& har_trans, unsigned int seed)
	: MinnSpillover(fit, lag_max, num_iter, num_burn, thin, ord, seed), har_trans(har_trans) {}
	// BvharSpillover(const MinnRecords& records, int lag_max, int ord, const Eigen::MatrixXd& har_trans)
	// : MinnSpillover(records, lag_max, ord), har_trans(har_trans) {}
	virtual ~BvharSpillover() = default;
	// void computeVma() override {
	// 	vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	// }
	void computeSpillover() override {
		for (size_t j = 0; j < record.size(); ++j) {
			vma_mat = convert_vhar_to_vma(record[j][0], har_trans, step - 1, lag);
			fevd += compute_vma_fevd(vma_mat, record[j][1], true);
		}
		fevd /= static_cast<int>(record.size());
		spillover = compute_sp_index(fevd);
	}
private:
	Eigen::MatrixXd har_trans;
};


}; // namespace bvhar

#endif // BVHAR_BAYES_MNIW_SPILLOVER_H