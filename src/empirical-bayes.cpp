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
  int num_design = object["obs"]; // s
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
