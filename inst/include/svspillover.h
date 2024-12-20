#ifndef SVSPILLOVER_H
#define SVSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

class SvSpillover;
class SvVharSpillover;

class SvSpillover : public McmcSpillover {
public:
	SvSpillover(const SvRecords& records, int lag_max, int ord, int id)
	: McmcSpillover(records, lag_max, ord, records.lvol_sig_record.cols(), id) {
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvSpillover() = default;
};

class SvVharSpillover : public SvSpillover {
public:
	SvVharSpillover(const SvRecords& records, int lag_max, int month, int id, const Eigen::MatrixXd& har_trans)
	: SvSpillover(records, lag_max, month, id), har_trans(har_trans) {}
	virtual ~SvVharSpillover() = default;

protected:
	void computeVma() override {
		vma_mat = convert_vhar_to_vma(coef_mat, har_trans, step - 1, lag);
	}

private:
	Eigen::MatrixXd har_trans; // without constant term
};

}; // namespace bvhar

#endif // SVSPILLOVER_H