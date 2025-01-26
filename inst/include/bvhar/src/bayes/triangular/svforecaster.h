#ifndef BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H

#include "./forecaster.h"

namespace bvhar {

// Until updating cpp sources
using SvVarForecaster = McmcVarForecaster<SvForecaster>;
using SvVharForecaster = McmcVharForecaster<SvForecaster>;
using SvVarSelectForecaster = McmcVarSelectForecaster<SvForecaster>;
using SvVharSelectForecaster = McmcVharSelectForecaster<SvForecaster>;

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H