#ifndef BVHAR_BAYES_DFM_CONFIG_H
#define BVHAR_BAYES_DFM_CONFIG_H

#include "../misc/draw.h"
#include "../../math/design.h"

namespace bvhar {

struct DfmParams;
struct DfmRecords;

struct DfmParams {
	int _size_factor, _lag, _dim;

	DfmParams(int lag, int dim_factor, int dim)
	: _size_factor(dim_factor), _lag(lag), _dim(dim) {}

	DfmParams(BVHAR_LIST& priors, int dim)
	: _size_factor(BVHAR_CAST_INT(priors["size_factor"])),
		_lag(BVHAR_CAST_INT(priors["lag"])),
		_dim(dim) {}
};

struct DfmRecords {
	Eigen::MatrixXd factor_record;

	DfmRecords(int num_iter, int num_design, int size_factor)
	: factor_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * size_factor)) {}

	DfmRecords(const Eigen::MatrixXd& factor_record)
	: factor_record(factor_record) {}
	
	virtual ~DfmRecords() = default;

	void assignRecords(
		int id, Eigen::Ref<Eigen::MatrixXd> factor_design,
		int num_design, int size_factor
	) {
		for (int i = 0; i < num_design; ++i) {
			factor_record.row(id).segment(i * size_factor, size_factor) = factor_design.col(i);
		}
	}

	virtual void appendRecords(BVHAR_LIST& list) {
		list["F_record"] = factor_record;
	}

	template <typename RecordType = DfmRecords>
	RecordType returnRecords(int num_iter, int num_burn, int thin) const;
};

template <>
inline DfmRecords DfmRecords::returnRecords(int num_iter, int num_burn, int thin) const {
	return DfmRecords(
		thin_record(factor_record, num_iter, num_burn, thin)
	);
}

inline void initialize_dfmrecord(
	std::unique_ptr<DfmRecords>& record, int chain_id, BVHAR_LIST& fit_record,
	BVHAR_STRING& f_name
) {
	BVHAR_PY_LIST f_list = fit_record[f_name];
	record = std::make_unique<DfmRecords>(BVHAR_CAST<Eigen::MatrixXd>(f_list[chain_id]));
}

} // namespace bvhar

#endif // BVHAR_BAYES_DFM_CONFIG_H