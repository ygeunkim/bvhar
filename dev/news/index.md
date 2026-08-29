# Changelog

## bvhar (development version)

- Fixed some wrong posteriors in NG and SSVS.

## bvhar 2.4.1

CRAN release: 2026-06-05

- Fixed Average LPL computation process in pseudo out-of-sample
  forecasting.

- Added IRF classes in C++.

### R

- Fixed wrong post-processing of MCMC posterior draws. This can change
  the results of post-analysis of the old code.

### Python

- `-DBVHAR_USE_PYBIND11` macro is required to use the header with
  Pybind11.

### C++

- Added boost.Random-based constructor for each MCMC initial values
  struct.

- Now header-only C++ library is available.

- Use [`Catch2`](https://github.com/catchorg/Catch2) for C++ unit tests.

## bvhar 2.4.0

CRAN release: 2026-02-01

### Important changes

- License has been changed to
  [MIT](https://choosealicense.com/licenses/mit/).

- `bvhar`-related packages now use `boost::random::mixmax` instead of
  `boost::random::mt19937` for pRNG, which is the same with Stan. This
  will affect every MCMC results of this library.

- Fixed DL prior’s latent parameter sampling.

- Fixed upper triangular Bartlett decomposition for inverse-Wishart rng.
  This change affects every results that uses MNIW in the posterior
  sampling.

- [`bvar_minnesota()`](../reference/bvar_minnesota.md) and
  [`bvhar_minnesota()`](../reference/bvhar_minnesota.md) will find
  hyperparameters in C++ using L-BFGS-B of `LBFGSpp` library. And these
  functions will be integrated into
  [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md) with new spec functions.
  Be careful when using these functions before the changes.

- `spdlog` loggers are replaced with custom progress bar (for master
  thread) by default. If defining `BVHAR_USE_SPDLOG`, spdlog will be
  used.

### New features

- [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md) can augment factor term
  with `factor_spec` argument.

- [`set_factor()`](../reference/set_factor.md) determines the factor
  size.

- [`predict()`](../reference/predict.md) can conduct in-sample
  forecasting when `n_ahead` is not specified (for now, only `*ldlt` and
  `*sv` can do this).

### Internal changes (C++)

- Remove `fmt::format()` which gave an error in C++20.

- Pseudo out-of-sample forecast classes have `isPath` template, which
  can give the path forecasting also in the rolling and expanding
  windows.

- Add `baecon` namespace wrapper around `bvhar` namespace in preparation
  for `baecon` verse-package integration. C++ users must update
  namespace references from `bvhar::` to `baecon::bvhar::`.

- Use forecaster classes with new `AutoregGenerator` class for VAR/VHAR
  generation.

- This gives different results due to boost RNG usage instead of R’s
  RNG.

- Add `BVHAR_` prefix to every defined macros. For example, `LIST` to
  `BVHAR_LIST`.

- Also, replace `USE_RCPP` and `USE_BVHAR_DEBUG` with `BVHAR_USE_RCPP`
  and `BVHAR_USE_BVHAR_DEBUG`.

- Add `use_fit` parameter to `McmcOutForecastRun` in the case when full
  sample analysis. If `use_fit = false`, the first window will be
  evaluated.

## bvhar 2.3.0

CRAN release: 2025-06-25

- Requires `R >= 4.2` due to Rtools 4.0 error with optional parameters
  in C++.

- Changed `bayes_spec` argument into `coef_spec` and `contem_spec` to
  enable different priors

- Added `mcmc` option in
  [`forecast_roll()`](../reference/forecast_roll.md) and
  [`forecast_expand()`](../reference/forecast_expand.md) for `ldltmod`
  and `svmod` classes.

### Exogenous variables

- [`var_lm()`](../reference/var_lm.md) and
  [`vhar_lm()`](../reference/vhar_lm.md) can run VARX and VHARX via
  `exogen` and `s`.

- [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md) can run Bayesian VARX and
  VHARX via `exogen`, `s`, and `exogen_spec`.

- `exogen_spec` determines the prior for the exogenous term.

- When forecasting these VARX and VHARX models,
  [`predict()`](../reference/predict.md) requires `newxreg`.

### Internal changes (C++)

- Added `shrinkage` headers for strategy design pattern.

- Added `McmcParams`, `McmcAlgo`, and `McmcRun` (changed original
  `McmcRun` to `CtaRun`) for extensibility of MCMC algorithms.

- Also added base forecaster classes.

- Changed the way OLS spillover classes work.

- Added `OlsSpilloverRun` and `OlsDynamicSpillover` for volatility
  spillover in OLS.

- If defining `USE_BVHAR_DEBUG` macro variable when compiling, users can
  see debug messages.

## bvhar 2.2.2

CRAN release: 2025-02-28

- Fix [`unlist()`](https://rdrr.io/r/base/unlist.html) error in print
  methods for `r-devel` (4.5.0).

## bvhar 2.2.1

CRAN release: 2025-02-25

## bvhar 2.2.0

CRAN release: 2025-02-06

- Requires `R >= 4.1` following [tidyverse R version support
  schedule](https://tidyverse.org/blog/2019/04/r-version-support/)

- `stable = TRUE` can filter MCMC draws where coefficient is stable when
  forecasting.

- Changed Eigen and boost assertion behavior (`eigen_assert` and
  `BOOST_ASSERT`) to give error instead of abort.

- Transpose the predictive distribution update loop.

- `med = TRUE` gives median of forecast draws as point forecast.

- [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md) can choose to use only
  group shrinkage parameters without global parameter with `ggl = FALSE`
  option.

- [`set_gdp()`](../reference/set_gdp.md) can use Generalized Double
  Pareto (GDP) shrinkage prior.

- [`alpl()`](../reference/alpl.md) gives summary of LPL across every
  horizon.

#### Internal changes

- Apply Devroye (2014) to draw GIG instead of Hörmann and Leydold.

- Use `spdlog` (using `RcppSpdlog`) logger instead of custom progress
  bar (`bvharprogress`).

- Use `RcppThread` to make the logger thread-safe
  ([eddelbuettel/rcppspdlog#22](https://github.com/eddelbuettel/rcppspdlog/issues/22))

- Use inverse-gamma prior for group parameters in DL.

- SAVS penalty is zero in own-lag.

#### Removal or deprecation

- Removed `sim_gig()` R function.

### C++ Header file changes

- Use template to avoid code duplicates among LDLT and SV models.

- Can easily conduct MCMC using `McmcRun` class in C++ source.

- Can easily implement forecasting for LDLT and SV MCMC using
  `McmcVarforecastRun<>` and `McmcVharforecastRun<>`.

- Can easily use rolling and expanding forecast for LDLT ans SV MCMC
  using `McmcVarforecastRun<>` and `McmcVharforecastRun<>`.

### Removal of deprecated functions

- Removed bvar_sv() and bvhar_sv().

- Removed bvar_ssvs(), bvhar_ssvs(), init_ssvs(), choose_ssvs(),
  sim_ssvs_var(), and sim_ssvs_vhar().

- Removed bvar_horseshoe(), bvhar_horseshoe(), sim_horseshoe_var(), and
  sim_horseshoe_vhar().

## bvhar 2.1.2

CRAN release: 2024-10-11

- Fix MCMC algorithm for `include_mean = TRUE` case.

- Fix predictive distribution update codes
  ([`predict()`](../reference/predict.md),
  [`forecast_roll()`](../reference/forecast_roll.md), and
  [`forecast_expand()`](../reference/forecast_expand.md) for `ldltmod`
  and `svmod` classes).

- Fix out-of-forecasting
  ([`forecast_roll()`](../reference/forecast_roll.md) and
  [`forecast_expand()`](../reference/forecast_expand.md)) result process
  codes.

## bvhar 2.1.1

CRAN release: 2024-10-05

- When using GIG generation in MCMC, it has maximum iteration numbers of
  while statement.

- Defined `USE_RCPP` macro in the C++ header so that Rcpp source usage
  works fine.

## bvhar 2.1.0

CRAN release: 2024-09-16

- Use Signal Adaptive Variable Selector (SAVS) to generate sparse
  coefficient from shrinkage priors.

- [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md) now handle both shrinkage
  priors and stochastic volatility.

- `bvar_ssvs()`, `bvar_horseshoe()`, `bvar_sv()`, `bvhar_ssvs()`,
  `bvhar_horseshoe()`, and `bvhar_sv()` are deprecated, and will be
  removed in v2.1.0 with their source functions.

- [`set_horseshoe()`](../reference/set_horseshoe.md) has additional
  setting for `group_shrinkage`. Horseshoe sampling now has additional
  group shrinkage level parameters.

- [`set_ssvs()`](../reference/set_ssvs.md) now additionally should
  specify different Beta hyperparameters for each own-lag and cross-lag.

- [`set_ssvs()`](../reference/set_ssvs.md) sets scaling factor and
  inverse-gamma hyperparameters for coefficients and cholesky factor
  slab sd.

- Use full bayesian approach to SSVS spike and slab sd’s instead of
  semi-automatic approach, in [`var_bayes()`](../reference/var_bayes.md)
  and [`vhar_bayes()`](../reference/vhar_bayes.md).

- MCMC functions return give `$param` and `$param_names`, not individual
  `$*_record` members.

- `sim_gig()` generates Generalized Inverse Gaussian (GIG) random
  numbers using the algorithm of R package `GIGrvg`.

### New priors

- [`set_dl()`](../reference/set_dl.md) specifies Dirichlet-Laplace (DL)
  prior in [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md).

- [`set_ng()`](../reference/set_ng.md) specifies Normal-Gamma (NG) prior
  in [`var_bayes()`](../reference/var_bayes.md) and
  [`vhar_bayes()`](../reference/vhar_bayes.md).

- `bvar_sv()` and `bvhar_sv()` supports hierarchical Minnesota prior.

### Internal changes

- Added regularization step in internal Normal posterior generation
  function against non-existing LLT case.

- Added `BOOST_DISABLE_ASSERTS` flag against `boost` asserts.

### Spillover effects

- [`spillover()`](../reference/spillover.md) computes static spillover
  given model.

- [`dynamic_spillover()`](../reference/dynamic_spillover.md) computes
  dynamic spillover given model.

### Forecasting

- [`predict()`](../reference/predict.md),
  [`forecast_roll()`](../reference/forecast_roll.md), and
  [`forecast_expand()`](../reference/forecast_expand.md) with LDLT
  models can use CI level when adding sparsity.

- [`predict()`](../reference/predict.md),
  [`forecast_roll()`](../reference/forecast_roll.md), and
  [`forecast_expand()`](../reference/forecast_expand.md) of `ldltmod`
  have `sparse` option to use sparsity.

- [`predict()`](../reference/predict.md),
  [`forecast_roll()`](../reference/forecast_roll.md), and
  [`forecast_expand()`](../reference/forecast_expand.md) with SV models
  can use CI level when adding sparsity.

- [`predict()`](../reference/predict.md),
  [`forecast_roll()`](../reference/forecast_roll.md), and
  [`forecast_expand()`](../reference/forecast_expand.md) of `svmod` have
  `sparse` option to use sparsity.

- Out-of-sample forecasting functions are now S3 generics
  ([`forecast_roll()`](../reference/forecast_roll.md) and
  [`forecast_expand()`](../reference/forecast_expand.md)).

- Add Rolling-window forecasting for LDLT models
  ([`forecast_roll.ldltmod()`](../reference/forecast_roll.md)).

- Add Expanding-window forecasting for LDLT models
  ([`forecast_expand.ldltmod()`](../reference/forecast_expand.md)).

- Add Rolling-window forecasting for SV models
  ([`forecast_roll.svmod()`](../reference/forecast_roll.md)).

- Add Expanding-window forecasting for SV models
  ([`forecast_expand.svmod()`](../reference/forecast_expand.md)).

- When forecasting SV models, it is available to choose whether to use
  time-varying covariance (`use_sv` option, which is `TRUE` by default).

- [`forecast_roll()`](../reference/forecast_roll.md) and
  [`forecast_expand()`](../reference/forecast_expand.md) can implement
  OpenMP multithreading, except in `bvarflat` class.

- If the model uses multiple chain MCMC, static schedule is used in
  [`forecast_roll()`](../reference/forecast_roll.md) and dynamic
  schedule in [`forecast_expand()`](../reference/forecast_expand.md).

- [`sim_mniw()`](../reference/sim_mniw.md) output format has been
  changed into list of lists.

- Now can use MNIW generation by including header
  (`std::vector<Eigen::MatrixXd> sim_mn_iw(...)`).

- Compute LPL inside
  [`forecast_roll.svmod()`](../reference/forecast_roll.md) and
  [`forecast_expand.svmod()`](../reference/forecast_expand.md) using
  `lpl` option.

- Instead, `lpl` method is removed.

## bvhar 2.0.1

CRAN release: 2024-03-01

- Fix internal vectorization and unvectorization behavior.

- Used Eigen 3.4 feature (`reshaped()`) to solve these
  (`RcppEigen >= 0.3.4.0.0`).

## bvhar 2.0.0

CRAN release: 2024-02-14

- Start to implement OOP in C++ source for each model, ready for major
  update.

- Add SV specification (`sv_spec` argument) in `bvhar_sv()` and
  `bvar_sv()` ([`set_sv()`](../reference/set_ldlt.md)).

- Prevent SSVS overflow issues by using log-sum-exp trick when computing
  Bernoulli posterior probability.

- Add separate constant term prior specification (`intercept`) in
  `bvhar_sv()` and `bvar_sv()`
  ([`set_intercept()`](../reference/set_intercept.md)).

- Convert every header file inst/include to header-only format. This
  enables external inclusion of our classes, structs, and Rcpp functions
  by using `LinkingTo` (in R package development) or
  `// [[Rcpp::depends(RcppEigen, BH, bvhar)]]`.

### Parallel Chain MCMC

- Use OpenMP parallel for loop

- Progress bar will show the status only for master thread when OpenMP
  enabled.

- Interruption detect will just save values and break the loop, not
  return immediately.

- Do burn-in and thinning in each `returnRecords()` method to make
  pre-process parallel chains easier.

- Use boost library (`BH` package) RNG instead of Rf\_\* RNG of `Rcpp`
  for thread-safety.

- Introduce function overloading to internal Rcpp random generation
  functions temporarily. It’s for maintaining
  [`set.seed()`](https://rdrr.io/r/base/Random.html) usage of some
  functions.

## bvhar 1.2.0

CRAN release: 2024-01-09

- Replace progress bar of `RcppProgress` package with custom header
  (`bvharprogress.h`).

- Replace checking user interruption in the same package with custom
  header (`bvharinterrupt.h`).

- Fix triangular algorithm. Found missing update of some variables
  (`bvar_sv()` and `bvhar_sv()`).

## bvhar 1.1.0

CRAN release: 2023-12-18

- For new research, add new features for shrinkage priors.

- Add Shrinkage priors SSVS and Horseshoe (`bvar_ssvs()`,
  `bvhar_ssvs()`, `bvar_horseshoe()`, and `bvhar_horseshoe()`).

- `bvar_sv()`, `bvhar_sv()` works with SSVS
  ([`set_ssvs()`](../reference/set_ssvs.md)) and Horseshoe
  ([`set_horseshoe()`](../reference/set_horseshoe.md)).

- Update the shrinkage structure in the spirit of Minnesota.
  (`minnesota = TRUE`, `minnesota = c("no", "short", "longrun")`).

- Stochastic volatility models implement corrected triangular algorithm
  of Carriero et al. (2021).

## bvhar 1.0.2

CRAN release: 2023-12-06

- License has been changed to
  [GPLv3](https://choosealicense.com/licenses/gpl-3.0/).

- Remove unnecessary Rcpp plugins in source files.

## bvhar 1.0.1

CRAN release: 2023-11-10

- Fix
  [`knitr::knit_print()`](https://rdrr.io/pkg/knitr/man/knit_print.html)
  method export methods
  [(](https://github.com/ygeunkim/bvhar/issues/2)[\#2](https://github.com/ygeunkim/bvhar/issues/2)).

## bvhar 1.0.0

CRAN release: 2023-11-08

- “Bayesian Vector Heterogeneous Autoregressive Modeling” has been
  accepted in JSCS 🎉

- Update to major version before publication.

## bvhar 0.14.1

## bvhar 0.14.0

- Add Stochastic Search Variable Selection (SSVS) models for VAR and
  VHAR (`bvar_ssvs()` and `bvhar_ssvs()`)

- Can do corresponding variable selection
  ([`summary.ssvsmod()`](../reference/summary.bvharsp.md))

## bvhar 0.13.0

- Add stochastic volatility models VAR-SV and VHAR-SV (`bvar_sv()` and
  `bvhar_sv()`).

## bvhar 0.12.1

- Fix not working Hierarchical natural conjugate MNIW function
  (`bvar_niwhm()`).

- Use `posterior` package for
  [`summary.normaliw()`](../reference/summary.normaliw.md) to improve
  processing and printing.

## bvhar 0.12.0

- Now can use heavy-tailed distribution ([Multivariate
  t-distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution))
  when generating VAR and VHAR process
  ([`sim_var()`](../reference/sim_var.md) and
  [`sim_vhar()`](../reference/sim_vhar.md)).

- Also provide independent MVT generation function
  ([`sim_mvt()`](../reference/sim_mvt.md)).

## bvhar 0.11.0

- Added `method = c("nor", "chol", "qr")` option in VAR and VHAR fitting
  function to use cholesky and Householder QR method
  ([`var_lm()`](../reference/var_lm.md) and
  [`vhar_lm()`](../reference/vhar_lm.md)).

- Now `include_mean` works internally with `Rcpp`.

## bvhar 0.10.0

- Add partial t-test for each VAR and VHAR coefficient
  ([`summary.varlse()`](../reference/summary.varlse.md) and
  [`summary.vharlse()`](../reference/summary.vharlse.md)).

- Appropriate print method for the updated summary method
  ([`print.summary.varlse()`](../reference/summary.varlse.md) and
  [`print.summary.vharlse()`](../reference/summary.vharlse.md)).

## bvhar 0.9.0

- Can compute impulse response function for VAR (`varlse`) and VHAR
  (`vharlse`) models (`analyze_ir()`).

- Can draw impulse -\> response plot in grid panels
  ([`autoplot.bvharirf()`](../reference/autoplot.bvharirf.md)).

## bvhar 0.8.0

- Changed the way of specifying the lower and upper bounds of empirical
  bayes ([`bound_bvhar()`](../reference/bound_bvhar.md)).

- Added Empirical Bayes vignette.

## bvhar 0.7.1

- When simulation, asymmetric covariance error is caught now
  (`sim_mgaussian()`).

## bvhar 0.7.0

- Add one integrated function that can do empirical bayes
  ([`choose_bayes()`](../reference/choose_bayes.md) and
  [`bound_bvhar()`](../reference/bound_bvhar.md)).

## bvhar 0.6.1

- Pre-process date column of `oxfordman` more elaborately (it becomes
  same with `etf_vix`).

## bvhar 0.6.0

- Added weekly and monthly order feature in VHAR family
  ([`vhar_lm()`](../reference/vhar_lm.md) and
  [`bvhar_minnesota()`](../reference/bvhar_minnesota.md)).

- Other functions are compatible with har order option
  ([`predict.vharlse()`](../reference/predict.md),
  [`predict.bvharmn()`](../reference/predict.md), and
  [`choose_bvhar()`](../reference/choose_bvar.md))

## bvhar 0.5.2

- Added parallel option for empirical bayes
  ([`choose_bvar()`](../reference/choose_bvar.md) and
  [`choose_bvhar()`](../reference/choose_bvar.md)).

## bvhar 0.5.1

- Added facet feature for the loss plot and changed its name
  ([`gg_loss()`](../reference/gg_loss.md)).

## bvhar 0.5.0

- Added rolling window and expanding window features
  ([`forecast_roll()`](../reference/forecast_roll.md) and
  [`forecast_expand()`](../reference/forecast_expand.md)).

- Can compute loss for each rolling and expanding window method
  ([`mse.bvharcv()`](../reference/mse.md),
  [`mae.bvharcv()`](../reference/mae.md),
  [`mape.bvharcv()`](../reference/mape.md), and
  [`mape.bvharcv()`](../reference/mape.md)).

## bvhar 0.4.1

- Fix Marginal likelihood form
  ([`compute_logml()`](../reference/compute_logml.md)).

- Optimize empirical bayes method using stabilized marginal likelihood
  function (`logml_stable()`).

## bvhar 0.4.0

- Change the way to compute the CI of BVAR and BVHAR
  ([`predict.bvarmn()`](../reference/predict.md),
  [`predict.bvharmn()`](../reference/predict.md), and
  [`predict.bvarflat()`](../reference/predict.md))

- Used custom random generation function - MN, IW, and MNIW based on
  RcppEigen

## bvhar 0.3.0

- Added Bayesian model specification functions and class (`bvharspec`).

- Replaced hyperparameters with model specification in Bayesian models
  ([`bvar_minnesota()`](../reference/bvar_minnesota.md),
  [`bvar_flat()`](../reference/bvar_flat.md), and
  [`bvhar_minnesota()`](../reference/bvhar_minnesota.md)).

## bvhar 0.2.0

- Added constant term choice in each function
  ([`var_lm()`](../reference/var_lm.md),
  [`vhar_lm()`](../reference/vhar_lm.md),
  [`bvar_minnesota()`](../reference/bvar_minnesota.md),
  [`bvar_flat()`](../reference/bvar_flat.md), and
  [`bvhar_minnesota()`](../reference/bvhar_minnesota.md)).

## bvhar 0.1.0

- Added a `NEWS.md` file to track changes to the package.
