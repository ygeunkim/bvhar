#ifndef BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H

#include "./forecaster.h"

namespace baecon {
namespace bvhar {

// Until updating cpp sources
using SvVarForecaster = CtaVarForecaster<SvForecaster>;
using SvVharForecaster = CtaVharForecaster<SvForecaster>;
using SvVarSelectForecaster = CtaVarSelectForecaster<SvForecaster>;
using SvVharSelectForecaster = CtaVharSelectForecaster<SvForecaster>;

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_TRIANGULAR_SVFORECASTER_H