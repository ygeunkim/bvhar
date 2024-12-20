#ifndef SVSPILLOVER_H
#define SVSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

class SvSpillover;
class SvVharSpillover;

class SvSpillover : public McmcVarSpillover {
public:
	SvSpillover(const SvRecords& records, int lag_max, int ord, int id)
	: McmcVarSpillover(records, lag_max, ord, records.lvol_sig_record.cols(), id) {
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvSpillover() = default;
};

class SvVharSpillover : public McmcVharSpillover {
public:
	SvVharSpillover(const SvRecords& records, int lag_max, int month, int id, const Eigen::MatrixXd& har_trans)
	: McmcVharSpillover(records, lag_max, month, records.lvol_sig_record.cols(), har_trans, id) {
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvVharSpillover() = default;
};

}; // namespace bvhar

#endif // SVSPILLOVER_H