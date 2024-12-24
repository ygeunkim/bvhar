#ifndef REGSPILLOVER_H
#define REGSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

using RegSpillover = McmcVarSpillover<LdltRecords>;
using RegVharSpillover = McmcVharSpillover<LdltRecords>;

}; // namespace bvhar

#endif // REGSPILLOVER_H