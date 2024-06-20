#include "bvhardraw.h"

//' Log of Multivariate Gamma Function
//' 
//' Compute log of multivariate gamma function numerically
//' 
//' @param x Double, non-negative argument
//' @param p Integer, dimension
//' @noRd
// [[Rcpp::export]]
double log_mgammafn(double x, int p) {
  if (p < 1) {
    Rcpp::stop("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (p == 1) {
    return bvhar::lgammafn(x);
  }
  if (2 * x < p) {
    Rcpp::stop("'x / 2' should be larger than 'p'.");
  }
  return bvhar::lmgammafn(x, p);
}

//' Numerically Stable Log ML Excluding Constant Term of BVAR and BVHAR
//' 
//' This function computes log of ML stable,
//' in purpose of objective function.
//' 
//' @param object Bayesian Model Fit
//' 
//' @noRd
// [[Rcpp::export]]
double logml_stable(Rcpp::List object) {
  if (!object.inherits("bvarmn") && !object.inherits("bvharmn")) {
    Rcpp::stop("'object' must be bvarmn or bvharmn object.");
  }
  return bvhar::compute_logml(object["m"], object["obs"], object["prior_precision"], object["prior_scale"], object["mn_prec"], object["covmat"], object["iw_shape"]);
}

//' AIC of VAR(p) using RSS
//' 
//' Compute AIC using RSS
//' 
//' @param object `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_aic(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = (2 / s) * number of freely estimated parameters
  return log(sig_det) + 2 / num_design * dim * dim_design;
}

//' BIC of VAR(p) using RSS
//' 
//' Compute BIC using RSS
//' 
//' @param object `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_bic(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = replace 2 / s with log(s) / s
  return log(sig_det) + log(num_design) / num_design * dim * dim_design;
}

//' HQ of VAR(p) using RSS
//' 
//' Compute HQ using RSS
//' 
//' @param object `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_hq(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = replace log(s) / s with 2 * log(log(s)) / s
  return log(sig_det) + 2 * log(log(num_design)) / num_design * dim * dim_design;
}

//' FPE of VAR(p) using RSS
//' 
//' Compute FPE using RSS
//' 
//' @param object `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_fpe(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  // FPE = ((s + k) / (s - k))^m * det = ((s + k) / s)^m * det(crossprod(resid) / (s - k))
  return pow((num_design + dim_design) / num_design, dim) * cov_lse.determinant();
}

//' Choose the Best VAR based on Information Criteria
//' 
//' This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR model.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag_max Maximum Var lag to explore
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd tune_var(Eigen::MatrixXd y, int lag_max, bool include_mean) {
  Rcpp::Function fit("var_lm");
  Eigen::MatrixXd ic_res(lag_max, 4); // matrix including information criteria: AIC-BIC-HQ-FPE
  Rcpp::List var_mod;
  for (int i = 0; i < lag_max; i++) {
    var_mod = fit(y, i + 1, include_mean);
    ic_res(i, 0) = compute_aic(var_mod);
    ic_res(i, 1) = compute_bic(var_mod);
    ic_res(i, 2) = compute_hq(var_mod);
    ic_res(i, 3) = compute_fpe(var_mod);
  }
  return ic_res;
}
