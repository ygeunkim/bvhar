# Package index

## Package

The bvhar package

- [`bvhar`](bvhar-package.md) [`bvhar-package`](bvhar-package.md) :
  bvhar: Bayesian Vector Heterogeneous Autoregressive Modeling

## Frequentist modeling

Vector autoregressive (VAR) and heterogeneous autoregressive (VHAR)
models.

- [`var_lm()`](var_lm.md) [`print(`*`<varlse>`*`)`](var_lm.md)
  [`logLik(`*`<varlse>`*`)`](var_lm.md)
  [`AIC(`*`<varlse>`*`)`](var_lm.md) [`BIC(`*`<varlse>`*`)`](var_lm.md)
  [`is.varlse()`](var_lm.md) [`is.bvharmod()`](var_lm.md)
  [`knit_print(`*`<varlse>`*`)`](var_lm.md) : Fitting Vector
  Autoregressive Model of Order p Model
- [`vhar_lm()`](vhar_lm.md) [`print(`*`<vharlse>`*`)`](vhar_lm.md)
  [`logLik(`*`<vharlse>`*`)`](vhar_lm.md)
  [`AIC(`*`<vharlse>`*`)`](vhar_lm.md)
  [`BIC(`*`<vharlse>`*`)`](vhar_lm.md) [`is.vharlse()`](vhar_lm.md)
  [`knit_print(`*`<vharlse>`*`)`](vhar_lm.md) : Fitting Vector
  Heterogeneous Autoregressive Model
- [`VARtoVMA()`](VARtoVMA.md) : Convert VAR to VMA(infinite)
- [`VHARtoVMA()`](VHARtoVMA.md) : Convert VHAR to VMA(infinite)
- [`summary(`*`<varlse>`*`)`](summary.varlse.md)
  [`print(`*`<summary.varlse>`*`)`](summary.varlse.md)
  [`knit_print(`*`<summary.varlse>`*`)`](summary.varlse.md) :
  Summarizing Vector Autoregressive Model
- [`summary(`*`<vharlse>`*`)`](summary.vharlse.md)
  [`print(`*`<summary.vharlse>`*`)`](summary.vharlse.md)
  [`knit_print(`*`<summary.vharlse>`*`)`](summary.vharlse.md) :
  Summarizing Vector HAR Model

## Prior specification

Prior settings for Bayesian models.

- [`set_bvar()`](set_bvar.md) [`set_bvar_flat()`](set_bvar.md)
  [`set_bvhar()`](set_bvar.md) [`set_weight_bvhar()`](set_bvar.md)
  [`print(`*`<bvharspec>`*`)`](set_bvar.md)
  [`is.bvharspec()`](set_bvar.md)
  [`knit_print(`*`<bvharspec>`*`)`](set_bvar.md) : Hyperparameters for
  Bayesian Models
- [`set_ssvs()`](set_ssvs.md) [`print(`*`<ssvsinput>`*`)`](set_ssvs.md)
  [`is.ssvsinput()`](set_ssvs.md)
  [`knit_print(`*`<ssvsinput>`*`)`](set_ssvs.md) : Stochastic Search
  Variable Selection (SSVS) Hyperparameter for Coefficients Matrix and
  Cholesky Factor
- [`set_lambda()`](set_lambda.md) [`set_psi()`](set_lambda.md)
  [`print(`*`<bvharpriorspec>`*`)`](set_lambda.md)
  [`is.bvharpriorspec()`](set_lambda.md)
  [`knit_print(`*`<bvharpriorspec>`*`)`](set_lambda.md) : Hyperpriors
  for Bayesian Models
- [`set_horseshoe()`](set_horseshoe.md)
  [`print(`*`<horseshoespec>`*`)`](set_horseshoe.md)
  [`is.horseshoespec()`](set_horseshoe.md)
  [`knit_print(`*`<horseshoespec>`*`)`](set_horseshoe.md) : Horseshoe
  Prior Specification
- [`set_ng()`](set_ng.md) [`print(`*`<ngspec>`*`)`](set_ng.md)
  [`is.ngspec()`](set_ng.md) **\[experimental\]** : Normal-Gamma
  Hyperparameter for Coefficients and Contemporaneous Coefficients
- [`set_dl()`](set_dl.md) [`print(`*`<dlspec>`*`)`](set_dl.md)
  [`is.dlspec()`](set_dl.md) **\[experimental\]** : Dirichlet-Laplace
  Hyperparameter for Coefficients and Contemporaneous Coefficients
- [`set_gdp()`](set_gdp.md) [`is.gdpspec()`](set_gdp.md)
  **\[experimental\]** : Generalized Double Pareto Shrinkage
  Hyperparameters for Coefficients and Contemporaneous Coefficients
- [`set_ldlt()`](set_ldlt.md) [`set_sv()`](set_ldlt.md)
  [`print(`*`<covspec>`*`)`](set_ldlt.md) [`is.covspec()`](set_ldlt.md)
  [`is.svspec()`](set_ldlt.md) [`is.ldltspec()`](set_ldlt.md)
  **\[experimental\]** : Covariance Matrix Prior Specification
- [`set_factor()`](set_factor.md) **\[experimental\]** : Factor
  Specification
- [`set_intercept()`](set_intercept.md)
  [`print(`*`<interceptspec>`*`)`](set_intercept.md)
  [`is.interceptspec()`](set_intercept.md)
  [`knit_print(`*`<interceptspec>`*`)`](set_intercept.md) : Prior for
  Constant Term

## Bayesian modeling

Bayesian VAR and Bayesian VHAR models.

- [`var_bayes()`](var_bayes.md) [`print(`*`<bvarsv>`*`)`](var_bayes.md)
  [`print(`*`<bvarldlt>`*`)`](var_bayes.md)
  [`knit_print(`*`<bvarsv>`*`)`](var_bayes.md)
  [`knit_print(`*`<bvarldlt>`*`)`](var_bayes.md) **\[maturing\]** :
  Fitting Bayesian VAR with Coefficient and Covariance Prior
- [`bvar_minnesota()`](bvar_minnesota.md)
  [`print(`*`<bvarmn>`*`)`](bvar_minnesota.md)
  [`print(`*`<bvarhm>`*`)`](bvar_minnesota.md)
  [`logLik(`*`<bvarmn>`*`)`](bvar_minnesota.md)
  [`AIC(`*`<bvarmn>`*`)`](bvar_minnesota.md)
  [`BIC(`*`<bvarmn>`*`)`](bvar_minnesota.md)
  [`is.bvarmn()`](bvar_minnesota.md)
  [`knit_print(`*`<bvarmn>`*`)`](bvar_minnesota.md)
  [`knit_print(`*`<bvarhm>`*`)`](bvar_minnesota.md) : Fitting Bayesian
  VAR(p) of Minnesota Prior
- [`bvar_flat()`](bvar_flat.md)
  [`print(`*`<bvarflat>`*`)`](bvar_flat.md)
  [`logLik(`*`<bvarflat>`*`)`](bvar_flat.md)
  [`AIC(`*`<bvarflat>`*`)`](bvar_flat.md)
  [`BIC(`*`<bvarflat>`*`)`](bvar_flat.md)
  [`is.bvarflat()`](bvar_flat.md)
  [`knit_print(`*`<bvarflat>`*`)`](bvar_flat.md) : Fitting Bayesian
  VAR(p) of Flat Prior
- [`vhar_bayes()`](vhar_bayes.md)
  [`print(`*`<bvharsv>`*`)`](vhar_bayes.md)
  [`print(`*`<bvharldlt>`*`)`](vhar_bayes.md)
  [`knit_print(`*`<bvharsv>`*`)`](vhar_bayes.md)
  [`knit_print(`*`<bvharldlt>`*`)`](vhar_bayes.md) **\[maturing\]** :
  Fitting Bayesian VHAR with Coefficient and Covariance Prior
- [`bvhar_minnesota()`](bvhar_minnesota.md)
  [`print(`*`<bvharmn>`*`)`](bvhar_minnesota.md)
  [`print(`*`<bvharhm>`*`)`](bvhar_minnesota.md)
  [`logLik(`*`<bvharmn>`*`)`](bvhar_minnesota.md)
  [`AIC(`*`<bvharmn>`*`)`](bvhar_minnesota.md)
  [`BIC(`*`<bvharmn>`*`)`](bvhar_minnesota.md)
  [`is.bvharmn()`](bvhar_minnesota.md)
  [`knit_print(`*`<bvharmn>`*`)`](bvhar_minnesota.md)
  [`knit_print(`*`<bvharhm>`*`)`](bvhar_minnesota.md) : Fitting Bayesian
  VHAR of Minnesota Prior
- [`summary(`*`<normaliw>`*`)`](summary.normaliw.md)
  [`print(`*`<summary.normaliw>`*`)`](summary.normaliw.md)
  [`knit_print(`*`<summary.normaliw>`*`)`](summary.normaliw.md) :
  Summarizing Bayesian Multivariate Time Series Model
- [`print(`*`<summary.bvharsp>`*`)`](summary.bvharsp.md)
  [`knit_print(`*`<summary.bvharsp>`*`)`](summary.bvharsp.md)
  [`summary(`*`<ssvsmod>`*`)`](summary.bvharsp.md)
  [`summary(`*`<hsmod>`*`)`](summary.bvharsp.md)
  [`summary(`*`<ngmod>`*`)`](summary.bvharsp.md) : Summarizing BVAR and
  BVHAR with Shrinkage Priors

## Deprecated

Deprecated functions that will be removed soon

- [`choose_bvar()`](choose_bvar.md) [`choose_bvhar()`](choose_bvar.md)
  [`print(`*`<bvharemp>`*`)`](choose_bvar.md)
  [`is.bvharemp()`](choose_bvar.md)
  [`knit_print(`*`<bvharemp>`*`)`](choose_bvar.md) **\[deprecated\]** :
  Finding the Set of Hyperparameters of Individual Bayesian Model
- [`bound_bvhar()`](bound_bvhar.md)
  [`print(`*`<boundbvharemp>`*`)`](bound_bvhar.md)
  [`is.boundbvharemp()`](bound_bvhar.md)
  [`knit_print(`*`<boundbvharemp>`*`)`](bound_bvhar.md)
  **\[deprecated\]** : Setting Empirical Bayes Optimization Bounds
- [`choose_bayes()`](choose_bayes.md) **\[deprecated\]** : Finding the
  Set of Hyperparameters of Bayesian Model

## Forecasting

- [`predict(`*`<varlse>`*`)`](predict.md)
  [`predict(`*`<vharlse>`*`)`](predict.md)
  [`predict(`*`<bvarmn>`*`)`](predict.md)
  [`predict(`*`<bvharmn>`*`)`](predict.md)
  [`predict(`*`<bvarflat>`*`)`](predict.md)
  [`predict(`*`<bvarldlt>`*`)`](predict.md)
  [`predict(`*`<bvharldlt>`*`)`](predict.md)
  [`predict(`*`<bvarsv>`*`)`](predict.md)
  [`predict(`*`<bvharsv>`*`)`](predict.md)
  [`print(`*`<predbvhar>`*`)`](predict.md)
  [`is.predbvhar()`](predict.md)
  [`knit_print(`*`<predbvhar>`*`)`](predict.md) : Forecasting
  Multivariate Time Series
- [`divide_ts()`](divide_ts.md) : Split a Time Series Dataset into
  Train-Test Set
- [`forecast_roll()`](forecast_roll.md)
  [`print(`*`<bvharcv>`*`)`](forecast_roll.md)
  [`is.bvharcv()`](forecast_roll.md)
  [`knit_print(`*`<bvharcv>`*`)`](forecast_roll.md) : Out-of-sample
  Forecasting based on Rolling Window
- [`forecast_expand()`](forecast_expand.md) : Out-of-sample Forecasting
  based on Expanding Window

## Structural analysis

- [`print(`*`<bvharirf>`*`)`](irf.md) [`irf()`](irf.md)
  [`is.bvharirf()`](irf.md) [`knit_print(`*`<bvharirf>`*`)`](irf.md) :
  Impulse Response Analysis
- [`spillover()`](spillover.md)
  [`print(`*`<bvharspillover>`*`)`](spillover.md)
  [`knit_print(`*`<bvharspillover>`*`)`](spillover.md) : h-step ahead
  Normalized Spillover
- [`dynamic_spillover()`](dynamic_spillover.md)
  [`print(`*`<bvhardynsp>`*`)`](dynamic_spillover.md)
  [`knit_print(`*`<bvhardynsp>`*`)`](dynamic_spillover.md) : Dynamic
  Spillover

## Evaluation

- [`mse()`](mse.md) : Evaluate the Model Based on MSE (Mean Square
  Error)
- [`mae()`](mae.md) : Evaluate the Model Based on MAE (Mean Absolute
  Error)
- [`mape()`](mape.md) : Evaluate the Model Based on MAPE (Mean Absolute
  Percentage Error)
- [`mase()`](mase.md) : Evaluate the Model Based on MASE (Mean Absolute
  Scaled Error)
- [`mrae()`](mrae.md) : Evaluate the Model Based on MRAE (Mean Relative
  Absolute Error)
- [`alpl()`](alpl.md) : Evaluate the Density Forecast Based on Average
  Log Predictive Likelihood (APLP)
- [`relmae()`](relmae.md) : Evaluate the Model Based on RelMAE (Relative
  MAE)
- [`rmsfe()`](rmsfe.md) : Evaluate the Model Based on RMSFE
- [`rmafe()`](rmafe.md) : Evaluate the Model Based on RMAFE
- [`rmape()`](rmape.md) : Evaluate the Model Based on RMAPE (Relative
  MAPE)
- [`rmase()`](rmase.md) : Evaluate the Model Based on RMASE (Relative
  MASE)
- [`conf_fdr()`](conf_fdr.md) : Evaluate the Sparsity Estimation Based
  on FDR
- [`conf_fnr()`](conf_fnr.md) : Evaluate the Sparsity Estimation Based
  on FNR
- [`conf_fscore()`](conf_fscore.md) : Evaluate the Sparsity Estimation
  Based on F1 Score
- [`conf_prec()`](conf_prec.md) : Evaluate the Sparsity Estimation Based
  on Precision
- [`conf_recall()`](conf_recall.md) : Evaluate the Sparsity Estimation
  Based on Recall
- [`confusion()`](confusion.md) : Evaluate the Sparsity Estimation Based
  on Confusion Matrix
- [`fromse()`](fromse.md) : Evaluate the Estimation Based on Frobenius
  Norm
- [`spne()`](spne.md) : Evaluate the Estimation Based on Spectral Norm
  Error
- [`relspne()`](relspne.md) : Evaluate the Estimation Based on Relative
  Spectral Norm Error
- [`compute_logml()`](compute_logml.md) : Extracting Log of Marginal
  Likelihood
- [`FPE()`](FPE.md) : Final Prediction Error Criterion
- [`HQ()`](HQ.md) : Hannan-Quinn Criterion
- [`choose_var()`](choose_var.md) : Choose the Best VAR based on
  Information Criteria
- [`compute_dic()`](compute_dic.md) : Deviance Information Criterion of
  Multivariate Time Series Model

## Plots

- [`autoplot(`*`<normaliw>`*`)`](autoplot.normaliw.md) : Residual Plot
  for Minnesota Prior VAR Model
- [`autoplot(`*`<summary.normaliw>`*`)`](autoplot.summary.normaliw.md) :
  Density Plot for Minnesota Prior VAR Model
- [`autoplot(`*`<predbvhar>`*`)`](autoplot.predbvhar.md)
  [`autolayer(`*`<predbvhar>`*`)`](autoplot.predbvhar.md) : Plot
  Forecast Result
- [`geom_eval()`](geom_eval.md) : Adding Test Data Layer
- [`gg_loss()`](gg_loss.md) : Compare Lists of Models
- [`autoplot(`*`<bvharirf>`*`)`](autoplot.bvharirf.md) : Plot Impulse
  Responses
- [`autoplot(`*`<bvharsp>`*`)`](autoplot.bvharsp.md) : Plot the Result
  of BVAR and BVHAR MCMC
- [`autoplot(`*`<summary.bvharsp>`*`)`](autoplot.summary.bvharsp.md) :
  Plot the Heatmap of SSVS Coefficients
- [`autoplot(`*`<bvhardynsp>`*`)`](autoplot.bvhardynsp.md) : Dynamic
  Spillover Indices Plot

## Simulation and Random Generation

- [`sim_var()`](sim_var.md) : Generate Multivariate Time Series Process
  Following VAR(p)
- [`sim_vhar()`](sim_vhar.md) : Generate Multivariate Time Series
  Process Following VAR(p)
- [`sim_mncoef()`](sim_mncoef.md) : Generate Minnesota BVAR Parameters
- [`sim_mnvhar_coef()`](sim_mnvhar_coef.md) : Generate Minnesota BVAR
  Parameters
- [`sim_mnormal()`](sim_mnormal.md) : Generate Multivariate Normal
  Random Vector
- [`sim_matgaussian()`](sim_matgaussian.md) : Generate Matrix Normal
  Random Matrix
- [`sim_iw()`](sim_iw.md) : Generate Inverse-Wishart Random Matrix
- [`sim_mniw()`](sim_mniw.md) : Generate Normal-IW Random Family
- [`sim_mvt()`](sim_mvt.md) : Generate Multivariate t Random Vector

## Data

- [`etf_vix`](etf_vix.md) : CBOE ETF Volatility Index Dataset

## Other generic functions

- [`stableroot()`](stableroot.md) : Roots of characteristic polynomial
- [`is.stable()`](is.stable.md) : Stability of the process
- [`coef(`*`<varlse>`*`)`](coef.md) [`coef(`*`<vharlse>`*`)`](coef.md)
  [`coef(`*`<bvarmn>`*`)`](coef.md) [`coef(`*`<bvarflat>`*`)`](coef.md)
  [`coef(`*`<bvharmn>`*`)`](coef.md) [`coef(`*`<bvharsp>`*`)`](coef.md)
  [`coef(`*`<summary.bvharsp>`*`)`](coef.md) : Coefficient Matrix of
  Multivariate Time Series Models
- [`residuals(`*`<varlse>`*`)`](residuals.md)
  [`residuals(`*`<vharlse>`*`)`](residuals.md)
  [`residuals(`*`<bvarmn>`*`)`](residuals.md)
  [`residuals(`*`<bvarflat>`*`)`](residuals.md)
  [`residuals(`*`<bvharmn>`*`)`](residuals.md) : Residual Matrix from
  Multivariate Time Series Models
- [`fitted(`*`<varlse>`*`)`](fitted.md)
  [`fitted(`*`<vharlse>`*`)`](fitted.md)
  [`fitted(`*`<bvarmn>`*`)`](fitted.md)
  [`fitted(`*`<bvarflat>`*`)`](fitted.md)
  [`fitted(`*`<bvharmn>`*`)`](fitted.md) : Fitted Matrix from
  Multivariate Time Series Models
