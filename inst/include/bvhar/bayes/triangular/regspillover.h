#ifndef BVHAR_BAYES_TRIANGULAR_REGSPILLOVER_H
#define BVHAR_BAYES_TRIANGULAR_REGSPILLOVER_H

#include "./spillover.h"

namespace bvhar {

using RegSpillover = McmcVarSpillover<LdltRecords>;
using RegVharSpillover = McmcVharSpillover<LdltRecords>;

}; // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_REGSPILLOVER_H