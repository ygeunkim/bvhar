#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Numerically Stable Log Marginal Likelihood Excluding Constant Term
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
  double dim = object["m"]; // m
  double num_design = object["obs"]; // s
  // posterior
  Eigen::MatrixXd mn_prec = object["mn_prec"]; // posterior precision of MN
  Eigen::MatrixXd iw_scale = object["iw_scale"]; // posterior scale of IW = Sighat
  double posterior_shape = object["iw_shape"]; // posterior shape of IW = a0 + s
  // prior
  Eigen::MatrixXd prior_prec = object["prior_precision"]; // prior precision of MN = Omega0^(-1) = Xp^T Xp
  Eigen::MatrixXd prior_scale = object["prior_scale"]; // prior scale of IW = S0
  // cholesky decomposition
  Eigen::LLT<Eigen::MatrixXd> lltOfmn(prior_prec.inverse());
  Eigen::MatrixXd chol_mn = lltOfmn.matrixL();
  Eigen::MatrixXd stable_mat_a = chol_mn.transpose() * (mn_prec - prior_prec) * chol_mn;
  Eigen::LLT<Eigen::MatrixXd> lltOfiw(prior_scale.inverse());
  Eigen::MatrixXd chol_iw = lltOfiw.matrixL();
  Eigen::MatrixXd stable_mat_b = chol_iw.transpose() * (iw_scale - prior_scale) * chol_iw;
  // eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(stable_mat_a);
  Eigen::VectorXd a_eigen = es_a.eigenvalues();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_b(stable_mat_b);
  Eigen::VectorXd b_eigen = es_b.eigenvalues();
  // sum of log(1 + eigenvalues)
  double a_term = a_eigen.array().log1p().sum();
  double b_term = b_eigen.array().log1p().sum();
  // result
  return - num_design / 2.0 * log(prior_scale.determinant()) - dim / 2.0 * a_term - posterior_shape / 2.0 * b_term;
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

