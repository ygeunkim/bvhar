#ifndef REGSPILLOVER_H
#define REGSPILLOVER_H

#include "mcmcreg.h"
#include "bvharstructural.h"

namespace bvhar {

class RegSpillover {
public:
	RegSpillover(const LdltRecords& records, int lag_max, int ord)
	: step(lag_max), lag(ord), reg_record(records),
		dim(records.fac_record.cols()),
		num_coef(reg_record.coef_record.cols()),
		num_sim(reg_record.coef_record.rows()),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		// h_time_record(reg_record.lvol_record.middleCols(id * dim, dim)),
		// lvol_sqrt(Eigen::MatrixXd::Zero(dim, dim)),
		diag_sqrt(reg_record.fac_record.row(0).cwiseSqrt().asDiagonal()),
		sqrt_sig(Eigen::MatrixXd::Zero(dim, dim)),
		cov(Eigen::MatrixXd::Zero(dim, dim)),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)),
		fevd(Eigen::MatrixXd::Zero(dim * step, dim)),
		spillover(Eigen::MatrixXd::Zero(dim, dim)) {}
	virtual ~RegSpillover() = default;
	virtual void computeVma() {
		vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
	}
	void computeSpillover() {
		for (int j = 0; j < num_sim; ++j) {
			// lvol_sqrt = (h_time_record.row(j) / 2).array().exp().matrix().asDiagonal();
			// sqrt_sig = build_inv_lower(dim, reg_record.contem_coef_record.row(j)).triangularView<Eigen::UnitLower>().solve(lvol_sqrt);
			diag_sqrt = reg_record.fac_record.row(j).cwiseSqrt().asDiagonal();
			sqrt_sig = build_inv_lower(dim, reg_record.contem_coef_record.row(j)).triangularView<Eigen::UnitLower>().solve(diag_sqrt);
			cov = sqrt_sig * sqrt_sig.transpose();
			coef_mat = unvectorize(reg_record.coef_record.row(j).transpose(), dim);
			computeVma();
			fevd += compute_vma_fevd(vma_mat, cov, true);
		}
		fevd /= num_sim;
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
	int lag; // p of VAR or month of VHAR
	LdltRecords reg_record;
	int dim;
	int num_coef;
	int num_sim;
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	// Eigen::MatrixXd h_time_record; // h_t record
	// Eigen::MatrixXd lvol_sqrt; // D_t^(1 / 2)
	Eigen::MatrixXd diag_sqrt; // D_t^(1 / 2)
	Eigen::MatrixXd sqrt_sig; // L^(-1) D_t(1 / 2)
	Eigen::MatrixXd cov; // Sigma_t
	Eigen::MatrixXd vma_mat;
	Eigen::MatrixXd fevd;
	Eigen::MatrixXd spillover;
};

class RegVharSpillover : public RegSpillover {
public:
	RegVharSpillover(const LdltRecords& records, int lag_max, int month, const Eigen::MatrixXd& har_trans)
	: RegSpillover(records, lag_max, month), har_trans(har_trans) {}
	virtual ~RegVharSpillover() = default;
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	}
private:
	Eigen::MatrixXd har_trans; // without constant term
};

}; // namespace bvhar

#endif // REGSPILLOVER_H