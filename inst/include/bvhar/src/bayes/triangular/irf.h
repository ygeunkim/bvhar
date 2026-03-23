#ifndef BVHAR_BAYES_TRIANGULAR_IRF_H
#define BVHAR_BAYES_TRIANGULAR_IRF_H

#include "../irf.h"
#include "./triangular.h"

namespace baecon {
namespace bvhar {

class CtaIrf;
template <typename RecordType> class CtaVarIrf;
template <typename RecordType> class CtaVharIrf;
template <typename RecordType> class CtaIrfRun;

class CtaIrf : public McmcIrf {
public:
	CtaIrf(const RegRecords& records, int lag_max, int ord, int dim, bool orthogonal = true, int id = -1)
	: McmcIrf(lag_max, ord, records.coef_record.rows(), orthogonal),
		time_id(id), dim(dim), num_coef(records.coef_record.cols()),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		sqrt_sig(Eigen::MatrixXd::Zero(dim, dim)),
		// cov(Eigen::MatrixXd::Zero(dim, dim)),
		vma_mat(Eigen::MatrixXd::Zero(dim * step, dim)) {
		vma_record = Eigen::MatrixXd::Zero(dim * step, num_sim * dim);
	}

	virtual ~CtaIrf() = default;
	
	// void computeIrf() override {
	// 	for (int i = 0; i < num_sim; ++i) {
	// 		reg_record->updateDiag(i, time_id, sv_update);
	// 		sqrt_sig = build_inv_lower(
	// 			dim,
	// 			reg_record->contem_coef_record.row(i)
	// 		).triangularView<Eigen::UnitLower>().solve(sv_update.asDiagonal().toDenseMatrix());
	// 		cov = sqrt_sig * sqrt_sig.transpose();
	// 		coef_mat = unvectorize(reg_record->coef_record.row(i).transpose(), dim);
	// 		computeVma();
	// 		// fevd.middleCols(i * dim, dim) = compute_vma_fevd(vma_mat, cov, true);
	// 		vma_record.middleCols(i * dim, dim) = vma_mat;
	// 	}
	// }

protected:
	int time_id, dim, num_coef;
	std::unique_ptr<RegRecords> reg_record;
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::VectorXd sv_update; // D_t^(1 / 2)
	Eigen::MatrixXd sqrt_sig; // L^(-1) D_t(1 / 2)
	// Eigen::MatrixXd cov; // Sigma_t
	Eigen::MatrixXd vma_mat;

	void updateParams(const int i) override {
		reg_record->updateDiag(i, time_id, sv_update); // D^{1/2} -> Should fix to get D_{T + h}^(1/2) in SV
		sqrt_sig = build_inv_lower(
			dim,
			reg_record->contem_coef_record.row(i)
		).triangularView<Eigen::UnitLower>().solve(sv_update.asDiagonal().toDenseMatrix());
		// cov = sqrt_sig * sqrt_sig.transpose();
		coef_mat = unvectorize(reg_record->coef_record.row(i).transpose(), dim);
	}

	// virtual void computeVma() = 0;
};

template <typename RecordType = LdltRecords>
class CtaVarIrf : public CtaIrf {
public:
	CtaVarIrf(RecordType& records, int lag_max, int ord, bool orthogonal = true, int id = -1)
	: CtaIrf(records, lag_max, ord, records.getDim(), orthogonal, id) {
		reg_record = std::make_unique<RecordType>(records);
	}
	virtual ~CtaVarIrf() = default;

protected:
	void updateMovingAverage(const int i) override {
		vma_mat = convert_var_to_vma(coef_mat, lag, step - 1);
		if (orthogonal) {
			for (int j = 0; j < step; ++j) {
				// vma_mat.middleRows(j * dim, dim) = sqrt_sig.transpose() * vma_mat.middleRows(j * dim, dim);
				vma_mat.middleRows(j * dim, dim).applyOnTheLeft(sqrt_sig.transpose());
			}
		}
		vma_record.middleCols(i * dim, dim) = vma_mat;
	}
};

template <typename RecordType = LdltRecords>
class CtaVharIrf : public CtaIrf {
public:
	CtaVharIrf(RecordType& records, int lag_max, int month, const Eigen::MatrixXd& har_trans, int id = -1)
	: CtaIrf(records, lag_max, month, records.getDim(), id), har_trans(har_trans) {
		reg_record = std::make_unique<RecordType>(records);
	}

	CtaVharIrf(RecordType& records, int lag_max, int week, int month, int id = -1)
	: CtaIrf(records, lag_max, month, records.getDim(), id),
		har_trans(build_vhar(records.getDim(), week, month, false)) {
		reg_record = std::make_unique<RecordType>(records);
	}

	virtual ~CtaVharIrf() = default;

protected:
	void updateMovingAverage(const int i) override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
		if (orthogonal) {
			for (int j = 0; j < step; ++j) {
				// vma_mat.middleRows(j * dim, dim) = sqrt_sig.transpose() * vma_mat.middleRows(j * dim, dim);
				vma_mat.middleRows(j * dim, dim).applyOnTheLeft(sqrt_sig.transpose());
			}
		}
		vma_record.middleCols(i * dim, dim) = vma_mat;
	}

private:
	Eigen::MatrixXd har_trans; // without constant term
};

template <typename RecordType = LdltRecords>
inline std::unique_ptr<CtaIrf> initialize_ctairf(
	int chain_id, int lag, int step, BVHAR_LIST& fit_record, bool sparse, int id = -1,
	BVHAR_OPTIONAL<Eigen::MatrixXd> har_trans = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> week = BVHAR_NULLOPT
) {
	std::unique_ptr<RecordType> reg_record;
	BVHAR_STRING coef_name = (har_trans || week) ? (sparse ? "phi_sparse_record" : "phi_record") : (sparse ? "alpha_sparse_record" : "alpha_record");
	BVHAR_STRING a_name = sparse ? "a_sparse_record" : "a_record";
	BVHAR_STRING c_name = sparse ? "c_sparse_record" : "c_record";
	initialize_record(reg_record, chain_id, fit_record, false, coef_name, a_name, c_name);
	std::unique_ptr<CtaIrf> irf_ptr;
	if (har_trans) {
		irf_ptr = std::make_unique<CtaVharIrf<RecordType>>(*reg_record, step, lag, *har_trans, id);
	} else if (week) {
		irf_ptr = std::make_unique<CtaVharIrf<RecordType>>(*reg_record, step, *week, lag, id);
	} else {
		irf_ptr = std::make_unique<CtaVarIrf<RecordType>>(*reg_record, step, lag, id);
	}
	return irf_ptr;
}

template <typename RecordType = LdltRecords>
class CtaIrfRun : public McmcIrfRun {
public:
	CtaIrfRun(int num_chains, int lag, int step, BVHAR_LIST& fit_record, bool sparse, int nthreads)
	: McmcIrfRun(num_chains, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"CtaIrfRun Constructor: num_chains={}, lag={}, step={}, sparse={}, nthreads={}",
			num_chains, lag, step, sparse, nthreads
		);
		for (int i = 0; i < num_chains; ++i) {
			irf_ptr[i] = initialize_ctairf<RecordType>(i, lag, step, fit_record, sparse, -1);
		}
	}
	
	CtaIrfRun(int num_chains, int week, int month, int step, BVHAR_LIST& fit_record, bool sparse, int nthreads)
	: McmcIrfRun(num_chains, nthreads) {
		for (int i = 0; i < num_chains; ++i) {
			irf_ptr[i] = initialize_ctairf<RecordType>(i, month, step, fit_record, sparse, -1, BVHAR_NULLOPT, week);
		}
	}
	
	virtual ~CtaIrfRun() = default;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_TRIANGULAR_IRF_H