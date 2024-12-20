#ifndef REGSPILLOVER_H
#define REGSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

class RegSpillover;
class RegVharSpillover;

class RegSpillover : public McmcSpillover {
public:
	RegSpillover(const LdltRecords& records, int lag_max, int ord)
	: McmcSpillover(records, lag_max, ord, records.fac_record.cols()) {
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegSpillover() = default;
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