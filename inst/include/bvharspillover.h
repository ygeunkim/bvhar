#ifndef BVHARSPILLOVER_H
#define BVHARSPILLOVER_H

#include "bvharmcmc.h"
#include "bvharstructural.h"

namespace bvhar {

class McmcSpillover;
class McmcVarSpillover;
class McmcVharSpillover;

/**
 * @brief Spillover class for `McmcTriangular`
 * 
 */
class McmcSpillover {
public:
	McmcSpillover(const RegRecords& records, int lag_max, int ord, int dim, int id = 0)
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
			NAMED("connect") = CAST_MATRIX(spillover),
			NAMED("to") = CAST_VECTOR(to_spillover),
			NAMED("from") = CAST_VECTOR(from_spillover),
			NAMED("tot") = CAST_VECTOR(tot_spillover),
			NAMED("net") = CAST_VECTOR(to_spillover - from_spillover),
			NAMED("net_pairwise") = CAST_MATRIX(net_spillover)
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
 */
class McmcVarSpillover : public McmcSpillover {
public:
	McmcVarSpillover(const RegRecords& records, int lag_max, int ord, int dim, int id = 0)
	: McmcSpillover(records, lag_max, ord, dim, id) {}
	virtual ~McmcVarSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
	}
};

/**
 * @brief Spillover class for VHAR with `McmcTriangular`
 * 
 */
class McmcVharSpillover : public McmcSpillover {
public:
	McmcVharSpillover(const RegRecords& records, int lag_max, int month, int dim, const Eigen::MatrixXd& har_trans, int id = 0)
	: McmcSpillover(records, lag_max, month, dim, id), har_trans(har_trans) {}
	virtual ~McmcVharSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	}

private:
	Eigen::MatrixXd har_trans; // without constant term
};

} // namespace bvhar

#endif // BVHARSPILLOVER_H