# bvhar 0.12.0

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
