#ifndef BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H

#include "./forecaster.h"

namespace bvhar {

// Until updating cpp sources
using RegVarForecaster = McmcVarForecaster<RegForecaster>;
using RegVharForecaster = McmcVharForecaster<RegForecaster>;
using RegVarSelectForecaster = McmcVarSelectForecaster<RegForecaster>;
using RegVharSelectForecaster = McmcVharSelectForecaster<RegForecaster>;

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H