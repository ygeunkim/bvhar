#ifndef SVSPILLOVER_H
#define SVSPILLOVER_H

#include "bvharspillover.h"

namespace bvhar {

using SvSpillover = McmcVarSpillover<SvRecords>;
using SvVharSpillover = McmcVharSpillover<SvRecords>;

}; // namespace bvhar

#endif // SVSPILLOVER_H