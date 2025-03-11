# bvhar (development version)

* Added `mcmc` option in `forecast_roll()` and `forecast_expand()` for `ldltmod` and `svmod` classes.

# bvhar 2.2.2

* Fix `unlist()` error in print methods for `r-devel` (4.5.0).

# bvhar 2.2.1

# bvhar 2.2.0

* Requires `R >= 4.1` following [tidyverse R version support schedule](https://www.tidyverse.org/blog/2019/04/r-version-support/)

* `stable = TRUE` can filter MCMC draws where coefficient is stable when forecasting.

* Changed Eigen and boost assertion behavior (`eigen_assert` and `BOOST_ASSERT`) to give error instead of abort.

* Transpose the predictive distribution update loop.

* `med = TRUE` gives median of forecast draws as point forecast.

* `var_bayes()` and `vhar_bayes()` can choose to use only group shrinkage parameters without global parameter with `ggl = FALSE` option.

* `set_gdp()` can use Generalized Double Pareto (GDP) shrinkage prior.

* `alpl()` gives summary of LPL across every horizon.

### Internal changes

* Apply Devroye (2014) to draw GIG instead of Hörmann and Leydold.

* Use `spdlog` (using `RcppSpdlog`) logger instead of custom progress bar (`bvharprogress`).

* Use `RcppThread` to make the logger thread-safe ([eddelbuettel/rcppspdlog#22](https://github.com/eddelbuettel/rcppspdlog/issues/22))

* Use inverse-gamma prior for group parameters in DL.

* SAVS penalty is zero in own-lag.

### Removal or deprecation

* Removed `sim_gig()` R function.

## C++ Header file changes

* Use template to avoid code duplicates among LDLT and SV models.

* Can easily conduct MCMC using `McmcRun` class in C++ source.

* Can easily implement forecasting for LDLT and SV MCMC using `McmcVarforecastRun<>` and `McmcVharforecastRun<>`.

* Can easily use rolling and expanding forecast for LDLT ans SV MCMC using `McmcVarforecastRun<>` and `McmcVharforecastRun<>`.

## Removal of deprecated functions

* Removed bvar_sv() and bvhar_sv().

* Removed bvar_ssvs(), bvhar_ssvs(), init_ssvs(), choose_ssvs(), sim_ssvs_var(), and sim_ssvs_vhar().

* Removed bvar_horseshoe(), bvhar_horseshoe(), sim_horseshoe_var(), and sim_horseshoe_vhar().

# bvhar 2.1.2

* Fix MCMC algorithm for `include_mean = TRUE` case.

* Fix predictive distribution update codes (`predict()`, `forecast_roll()`, and `forecast_expand()` for `ldltmod` and `svmod` classes).

* Fix out-of-forecasting (`forecast_roll()` and `forecast_expand()`) result process codes.

# bvhar 2.1.1

* When using GIG generation in MCMC, it has maximum iteration numbers of while statement.

* Defined `USE_RCPP` macro in the C++ header so that Rcpp source usage works fine.

# bvhar 2.1.0

* Use Signal Adaptive Variable Selector (SAVS) to generate sparse coefficient from shrinkage priors.

* `var_bayes()` and `vhar_bayes()` now handle both shrinkage priors and stochastic volatility.

* `bvar_ssvs()`, `bvar_horseshoe()`, `bvar_sv()`, `bvhar_ssvs()`, `bvhar_horseshoe()`, and `bvhar_sv()` are deprecated, and will be removed in v2.1.0 with their source functions.

* `set_horseshoe()` has additional setting for `group_shrinkage`. Horseshoe sampling now has additional group shrinkage level parameters.

* `set_ssvs()` now additionally should specify different Beta hyperparameters for each own-lag and cross-lag.

* `set_ssvs()` sets scaling factor and inverse-gamma hyperparameters for coefficients and cholesky factor slab sd.

* Use full bayesian approach to SSVS spike and slab sd's instead of semi-automatic approach, in `var_bayes()` and `vhar_bayes()`.

* MCMC functions return give `$param` and `$param_names`, not individual `$*_record` members.

* `sim_gig()` generates Generalized Inverse Gaussian (GIG) random numbers using the algorithm of R package `GIGrvg`.

## New priors

* `set_dl()` specifies Dirichlet-Laplace (DL) prior in `var_bayes()` and `vhar_bayes()`.

* `set_ng()` specifies Normal-Gamma (NG) prior in `var_bayes()` and `vhar_bayes()`.

* `bvar_sv()` and `bvhar_sv()` supports hierarchical Minnesota prior.

## Internal changes

* Added regularization step in internal Normal posterior generation function against non-existing LLT case.

* Added `BOOST_DISABLE_ASSERTS` flag against `boost` asserts.

## Spillover effects

* `spillover()` computes static spillover given model.

* `dynamic_spillover()` computes dynamic spillover given model.

## Forecasting

* `predict()`, `forecast_roll()`, and `forecast_expand()` with LDLT models can use CI level when adding sparsity.

* `predict()`, `forecast_roll()`, and `forecast_expand()` of `ldltmod` have `sparse` option to use sparsity.

* `predict()`, `forecast_roll()`, and `forecast_expand()` with SV models can use CI level when adding sparsity.

* `predict()`, `forecast_roll()`, and `forecast_expand()` of `svmod` have `sparse` option to use sparsity.

* Out-of-sample forecasting functions are now S3 generics (`forecast_roll()` and `forecast_expand()`).

* Add Rolling-window forecasting for LDLT models (`forecast_roll.ldltmod()`).

* Add Expanding-window forecasting for LDLT models (`forecast_expand.ldltmod()`).

* Add Rolling-window forecasting for SV models (`forecast_roll.svmod()`).

* Add Expanding-window forecasting for SV models (`forecast_expand.svmod()`).

* When forecasting SV models, it is available to choose whether to use time-varying covariance (`use_sv` option, which is `TRUE` by default).

* `forecast_roll()` and `forecast_expand()` can implement OpenMP multithreading, except in `bvarflat` class.

* If the model uses multiple chain MCMC, static schedule is used in `forecast_roll()` and dynamic schedule in `forecast_expand()`.

* `sim_mniw()` output format has been changed into list of lists.

* Now can use MNIW generation by including header (`std::vector<Eigen::MatrixXd> sim_mn_iw(...)`).

* Compute LPL inside `forecast_roll.svmod()` and `forecast_expand.svmod()` using `lpl` option.

* Instead, `lpl` method is removed.

# bvhar 2.0.1

* Fix internal vectorization and unvectorization behavior.

* Used Eigen 3.4 feature (`reshaped()`) to solve these (`RcppEigen >= 0.3.4.0.0`).

# bvhar 2.0.0

* Start to implement OOP in C++ source for each model, ready for major update.

* Add SV specification (`sv_spec` argument) in `bvhar_sv()` and `bvar_sv()` (`set_sv()`).

* Prevent SSVS overflow issues by using log-sum-exp trick when computing Bernoulli posterior probability.

* Add separate constant term prior specification (`intercept`) in `bvhar_sv()` and `bvar_sv()` (`set_intercept()`).

* Convert every header file inst/include to header-only format. This enables external inclusion of our classes, structs, and Rcpp functions by using `LinkingTo` (in R package development) or `// [[Rcpp::depends(RcppEigen, BH, bvhar)]]`.

## Parallel Chain MCMC

* Use OpenMP parallel for loop

* Progress bar will show the status only for master thread when OpenMP enabled.

* Interruption detect will just save values and break the loop, not return immediately.

* Do burn-in and thinning in each `returnRecords()` method to make pre-process parallel chains easier.

* Use boost library (`BH` package) RNG instead of Rf_* RNG of `Rcpp` for thread-safety.

* Introduce function overloading to internal Rcpp random generation functions temporarily.
It's for maintaining `set.seed()` usage of some functions.

# bvhar 1.2.0

* Replace progress bar of `RcppProgress` package with custom header (`bvharprogress.h`).

* Replace checking user interruption in the same package with custom header (`bvharinterrupt.h`).

* Fix triangular algorithm. Found missing update of some variables (`bvar_sv()` and `bvhar_sv()`).

# bvhar 1.1.0

* For new research, add new features for shrinkage priors.

* Add Shrinkage priors SSVS and Horseshoe (`bvar_ssvs()`, `bvhar_ssvs()`, `bvar_horseshoe()`, and `bvhar_horseshoe()`).

* `bvar_sv()`, `bvhar_sv()` works with SSVS (`set_ssvs()`) and Horseshoe (`set_horseshoe()`).

* Update the shrinkage structure in the spirit of Minnesota. (`minnesota = TRUE`, `minnesota = c("no", "short", "longrun")`).

* Stochastic volatility models implement corrected triangular algorithm of Carriero et al. (2021).

# bvhar 1.0.2

* License has been changed to [GPLv3](https://choosealicense.com/licenses/gpl-3.0/).

* Remove unnecessary Rcpp plugins in source files.

# bvhar 1.0.1

* Fix `knitr::knit_print()` method export methods [(#2)](https://github.com/ygeunkim/bvhar/issues/2).

# bvhar 1.0.0

* "Bayesian Vector Heterogeneous Autoregressive Modeling" has been accepted in JSCS 🎉

* Update to major version before publication.

# bvhar 0.14.1

# bvhar 0.14.0

* Add Stochastic Search Variable Selection (SSVS) models for VAR and VHAR (`bvar_ssvs()` and `bvhar_ssvs()`)

* Can do corresponding variable selection (`summary.ssvsmod()`)

# bvhar 0.13.0

* Add stochastic volatility models VAR-SV and VHAR-SV (`bvar_sv()` and `bvhar_sv()`).

# bvhar 0.12.1

* Fix not working Hierarchical natural conjugate MNIW function (`bvar_niwhm()`).

* Use `posterior` package for `summary.normaliw()` to improve processing and printing.

# bvhar 0.12.0

* Now can use heavy-tailed distribution ([Multivariate t-distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution)) when generating VAR and VHAR process (`sim_var()` and `sim_vhar()`).

* Also provide independent MVT generation function (`sim_mvt()`).

# bvhar 0.11.0

* Added `method = c("nor", "chol", "qr")` option in VAR and VHAR fitting function to use cholesky and Householder QR method (`var_lm()` and `vhar_lm()`).

* Now `include_mean` works internally with `Rcpp`.

# bvhar 0.10.0

* Add partial t-test for each VAR and VHAR coefficient (`summary.varlse()` and `summary.vharlse()`).

* Appropriate print method for the updated summary method (`print.summary.varlse()` and `print.summary.vharlse()`).

# bvhar 0.9.0

* Can compute impulse response function for VAR (`varlse`) and VHAR (`vharlse`) models (`analyze_ir()`).

* Can draw impulse -> response plot in grid panels (`autoplot.bvharirf()`).

# bvhar 0.8.0

* Changed the way of specifying the lower and upper bounds of empirical bayes (`bound_bvhar()`).

* Added Empirical Bayes vignette.

# bvhar 0.7.1

* When simulation, asymmetric covariance error is caught now (`sim_mgaussian()`).

# bvhar 0.7.0

* Add one integrated function that can do empirical bayes (`choose_bayes()` and `bound_bvhar()`).

# bvhar 0.6.1

* Pre-process date column of `oxfordman` more elaborately (it becomes same with `etf_vix`).

# bvhar 0.6.0

* Added weekly and monthly order feature in VHAR family (`vhar_lm()` and `bvhar_minnesota()`).

* Other functions are compatible with har order option (`predict.vharlse()`, `predict.bvharmn()`, and `choose_bvhar()`)

# bvhar 0.5.2

* Added parallel option for empirical bayes (`choose_bvar()` and `choose_bvhar()`).

# bvhar 0.5.1

* Added facet feature for the loss plot and changed its name (`gg_loss()`).

# bvhar 0.5.0

* Added rolling window and expanding window features (`forecast_roll()` and `forecast_expand()`).

* Can compute loss for each rolling and expanding window method (`mse.bvharcv()`, `mae.bvharcv()`, `mape.bvharcv()`, and `mape.bvharcv()`).

# bvhar 0.4.1

* Fix Marginal likelihood form (`compute_logml()`).

* Optimize empirical bayes method using stabilized marginal likelihood function (`logml_stable()`).

# bvhar 0.4.0

* Change the way to compute the CI of BVAR and BVHAR (`predict.bvarmn()`, `predict.bvharmn()`, and `predict.bvarflat()`)

* Used custom random generation function - MN, IW, and MNIW based on RcppEigen

# bvhar 0.3.0

* Added Bayesian model specification functions and class (`bvharspec`).

* Replaced hyperparameters with model specification in Bayesian models (`bvar_minnesota()`, `bvar_flat()`, and `bvhar_minnesota()`).

# bvhar 0.2.0

* Added constant term choice in each function (`var_lm()`, `vhar_lm()`, `bvar_minnesota()`, `bvar_flat()`, and `bvhar_minnesota()`).

# bvhar 0.1.0

* Added a `NEWS.md` file to track changes to the package.
