#ifndef MCMCSV_H
#define MCMCSV_H

#include "bvharmcmc.h"

namespace bvhar {

// Until updating cpp sources
using MinnSv = McmcMinn<McmcSv>;
using HierminnSv = McmcHierminn<McmcSv>;
using SsvsSv = McmcSsvs<McmcSv>;
using HorseshoeSv = McmcHorseshoe<McmcSv>;
using NormalgammaSv = McmcNg<McmcSv>;
using DirLaplaceSv = McmcDl<McmcSv>;

} // namespace bvhar

#endif // MCMCSV_H