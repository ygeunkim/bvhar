#ifndef REGSPILLOVER_H
#define REGSPILLOVER_H

#include "mcmcreg.h"
#include "bvharstructural.h"

namespace bvhar {

class RegSpillover;
class RegVharSpillover;

class RegSpillover {
public:
	RegSpillover(const LdltRecords& records, int lag_max, int ord)
	: step(lag_max), lag(ord), reg_record(records),
		dim(records.fac_record.cols()), num_coef(reg_record.coef_record.cols()),
		num_sim(reg_record.coef_record.rows()),
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
		net_spillover(Eigen::MatrixXd::Zero(dim, num_sim * dim)) {
		reg_record.updateDiag(0, sv_update);
	}
	virtual ~RegSpillover() = default;
	void computeSpillover() {
		for (int i = 0; i < num_sim; ++i) {
			reg_record.updateDiag(0, sv_update);
			sqrt_sig = build_inv_lower(dim, reg_record.contem_coef_record.row(i)).triangularView<Eigen::UnitLower>().solve(sv_update.asDiagonal().toDenseMatrix());
			cov = sqrt_sig * sqrt_sig.transpose();
			coef_mat = unvectorize(reg_record.coef_record.row(i).transpose(), dim);
			computeVma();
			fevd.middleCols(i * dim, dim) = compute_vma_fevd(vma_mat, cov, true);
			spillover.middleCols(i * dim, dim) = compute_sp_index(fevd.middleCols(i * dim, dim));
			to_spillover.segment(i * dim, dim) = compute_to(spillover.middleCols(i * dim, dim));
			from_spillover.segment(i * dim, dim) = compute_from(spillover.middleCols(i * dim, dim));
			tot_spillover[i] = compute_tot(spillover.middleCols(i * dim, dim));
			net_spillover.middleCols(i * dim, dim) = compute_net(spillover.middleCols(i * dim, dim));
		}
	}
	LIST returnSpilloverDensity() {
		computeSpillover();
		LIST res = CREATE_LIST(
			NAMED("connect") = CAST_MATRIX(spillover),
			NAMED("to") = CAST_VECTOR(to_spillover),
			NAMED("from") = CAST_VECTOR(from_spillover),
			NAMED("tot") = CAST_VECTOR(tot_spillover),
			NAMED("net") = CAST_VECTOR(to_spillover - from_spillover),
			NAMED("net_pairwise") = CAST_MATRIX(net_spillover)
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
	// double returnTot() {
	// 	return compute_tot(spillover);
	// }
	Eigen::VectorXd returnTot() {
		return tot_spillover;
	}
	Eigen::MatrixXd returnNet() {
		return net_spillover;
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
	virtual void computeVma() {
		vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
	}
};

class RegVharSpillover : public RegSpillover {
public:
	RegVharSpillover(const LdltRecords& records, int lag_max, int month, const Eigen::MatrixXd& har_trans)
	: RegSpillover(records, lag_max, month), har_trans(har_trans) {}
	virtual ~RegVharSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	}

private:
	Eigen::MatrixXd har_trans; // without constant term
};

}; // namespace bvhar

#endif // REGSPILLOVER_H