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
