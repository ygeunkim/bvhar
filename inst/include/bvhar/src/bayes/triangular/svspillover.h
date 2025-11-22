#ifndef BVHAR_BAYES_TRIANGULAR_SVSPILLOVER_H
#define BVHAR_BAYES_TRIANGULAR_SVSPILLOVER_H

#include "./spillover.h"

namespace baecon {
namespace bvhar {

using SvSpillover = McmcVarSpillover<SvRecords>;
using SvVharSpillover = McmcVharSpillover<SvRecords>;

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_TRIANGULAR_SVSPILLOVER_H