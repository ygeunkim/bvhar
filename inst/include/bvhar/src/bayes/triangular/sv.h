#ifndef BVHAR_BAYES_TRIANGULAR_SV_H
#define BVHAR_BAYES_TRIANGULAR_SV_H

#include "./triangular.h"

namespace bvhar {

// Until updating cpp sources
using MinnSv = McmcMinn<McmcSv>;
using HierminnSv = McmcHierminn<McmcSv>;
using SsvsSv = McmcSsvs<McmcSv>;
using HorseshoeSv = McmcHorseshoe<McmcSv>;
using NormalgammaSv = McmcNg<McmcSv>;
using DirLaplaceSv = McmcDl<McmcSv>;

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_SV_H