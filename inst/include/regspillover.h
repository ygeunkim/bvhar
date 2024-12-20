#ifndef REGSPILLOVER_H
#define REGSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

class RegSpillover;
class RegVharSpillover;

class RegSpillover : public McmcVarSpillover {
public:
	RegSpillover(const LdltRecords& records, int lag_max, int ord)
	: McmcVarSpillover(records, lag_max, ord, records.fac_record.cols()) {
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegSpillover() = default;
};

class RegVharSpillover : public McmcVharSpillover {
public:
	RegVharSpillover(const LdltRecords& records, int lag_max, int month, const Eigen::MatrixXd& har_trans)
	: McmcVharSpillover(records, lag_max, month, records.fac_record.cols(), har_trans) {
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegVharSpillover() = default;
};

}; // namespace bvhar

#endif // REGSPILLOVER_H