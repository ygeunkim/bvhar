#ifndef REGFORECASTER_H
#define REGFORECASTER_H

#include "bvharforecaster.h"

namespace bvhar {

// Until updating cpp sources
using RegVarForecaster = McmcVarForecaster<RegForecaster>;
using RegVharForecaster = McmcVharForecaster<RegForecaster>;
using RegVarSelectForecaster = McmcVarSelectForecaster<RegForecaster>;
using RegVharSelectForecaster = McmcVharSelectForecaster<RegForecaster>;

} // namespace bvhar

#endif // REGFORECASTER_H