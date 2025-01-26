#ifndef BVHAR_BAYES_TRIANGULAR_REG_H
#define BVHAR_BAYES_TRIANGULAR_REG_H

#include "./triangular.h"

namespace bvhar {

// Until updating cpp sources
using MinnReg = McmcMinn<McmcReg>;
using HierminnReg = McmcHierminn<McmcReg>;
using SsvsReg = McmcSsvs<McmcReg>;
using HorseshoeReg = McmcHorseshoe<McmcReg>;
using NgReg = McmcNg<McmcReg>;
using DlReg = McmcDl<McmcReg>;

}; // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_REG_H