#ifndef SVFORECASTER_H
#define SVFORECASTER_H

#include "bvharforecaster.h"

namespace bvhar {

// Until updating cpp sources
using SvVarForecaster = McmcVarForecaster<SvForecaster>;
using SvVharForecaster = McmcVharForecaster<SvForecaster>;
using SvVarSelectForecaster = McmcVarSelectForecaster<SvForecaster>;
using SvVharSelectForecaster = McmcVharSelectForecaster<SvForecaster>;

} // namespace bvhar

#endif // SVFORECASTER_H