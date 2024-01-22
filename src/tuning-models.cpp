#include <RcppEigen.h>
#include <bvhardraw.h>

//' Numerically Stable Log Marginal Likelihood Excluding Constant Term
//' 
//' This function computes log of ML stable,
//' excluding the constant term.
//' 
//' @param dim Dimension of the time series
//' @param num_design The number of the data matrix, \eqn{n = T - p}
//' @param prior_prec Prior precision of Matrix Normal distribution
//' @param prior_scale Prior scale of Inverse-Wishart distribution
//' @param mn_prec Posterior precision of Matrix Normal distribution
//' @param iw_scale Posterior scale of Inverse-Wishart distribution
//' @param posterior_shape Posterior shape of Inverse-Wishart distribution
//' 
//' @noRd
// [[Rcpp::export]]
double compute_logml(int dim, 
                     int num_design,
                     Eigen::MatrixXd prior_prec,
                     Eigen::MatrixXd prior_scale,
                     Eigen::MatrixXd mn_prec,
                     Eigen::MatrixXd iw_scale,
                     int posterior_shape) {
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
  return compute_logml(object["m"], object["obs"], object["prior_precision"], object["prior_scale"], object["mn_prec"], object["iw_scale"], object["iw_shape"]);
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

//' log Density of Multivariate Normal with LDLT Precision Matrix
//' 
//' Compute log density of multivariate normal with LDLT precision matrix decomposition.
//' 
//' @param x Point
//' @param mean_vec Mean
//' @param lower_vec row of a_record
//' @param diag_vec row of h_record
//' 
//' @noRd
// [[Rcpp::export]]
double compute_log_dmgaussian(Eigen::VectorXd x,
                              Eigen::VectorXd mean_vec,
                              Eigen::VectorXd lower_vec,
                              Eigen::VectorXd diag_vec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd diag_mat = Eigen::MatrixXd::Zero(dim, dim); // sqrt(D) in LDLT
  diag_mat.diagonal() = 1 / diag_vec.array().exp().sqrt(); // exp since D = exp(h)
  Eigen::MatrixXd lower_mat = build_inv_lower(dim, lower_vec);
  x.array() -= mean_vec.array(); // x - mu
  Eigen::VectorXd y = diag_mat * lower_mat * x; // sqrt(D) * L * (x - mu)
  double res = -log(lower_vec.squaredNorm()) - log(diag_vec.sum()) / 2 - dim * log(2 * M_PI) / 2; // should fix this line?
  res -= y.squaredNorm() / 2;
  return res;
}

//' Compute Log Predictive Likelihood
//' 
//' This function computes log-predictive likelihood (LPL).
//' 
//' @param True value
//' @param Predicted value
//' @param h_last_record MCMC record of log-volatilities in last time
//' @param a_record MCMC record of contemporaneous coefficients
//' @param sigh_record MCMC record of variance of log-volatilities
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_lpl(Eigen::MatrixXd y,
                            Eigen::MatrixXd posterior_mean,
                            Eigen::MatrixXd h_last_record,
                            Eigen::MatrixXd a_record,
                            Eigen::MatrixXd sigh_record) {
  int num_sim = a_record.rows();
  int dim = h_last_record.cols();
  int num_pred = y.rows();
  Eigen::VectorXd lpl_hist = Eigen::VectorXd::Zero(num_pred);
  Eigen::VectorXd sv_update(dim);
  Eigen::MatrixXd sv_cov = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < num_pred; i++) {
    for (int b = 0; b < num_sim; b++) {
      sv_cov.diagonal() = 1 / sigh_record.row(b).array();
      sv_update = vectorize_eigen(
        sim_mgaussian_chol(1, h_last_record.row(b), sv_cov)
      );
      lpl_hist[i] += compute_log_dmgaussian(
        y.row(i),
        posterior_mean.row(i),
        a_record.row(b),
        sv_update
      );
    }
    lpl_hist[i] /= num_sim;
  }
  // return lpl_hist.mean();
  return lpl_hist;
}
