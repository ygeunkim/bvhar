#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' h-step ahead Forecast Error Variance Decomposition
//' 
//' [w_(h = 1, ij)^T, w_(h = 2, ij)^T, ...]
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize) {
    int dim = cov_mat.cols();
    // Eigen::MatrixXd vma_mat = VARcoeftoVMA(var_coef, var_lag, step);
    int step = vma_coef.rows() / dim; // h-step
    Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd ma_prod(dim, dim);
    Eigen::MatrixXd numer = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd denom = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * step, dim);
    Eigen::MatrixXd cov_diag = Eigen::MatrixXd::Zero(dim, dim);
    cov_diag.diagonal() = 1 / cov_mat.diagonal().cwiseSqrt().array(); // sigma_jj
    for (int i = 0; i < step; i++) {
        ma_prod = vma_coef.block(i * dim, 0, dim, dim).transpose() * cov_mat; // A * Sigma
        innov_account += ma_prod * vma_coef.block(i * dim, 0, dim, dim); // A * Sigma * A^T
        numer.array() += (ma_prod * cov_diag).array().square(); // sum(A * Sigma)_ij / sigma_jj^2
        denom.diagonal() = 1 / innov_account.diagonal().array(); // sigma_jj^(-1) / sum(A * Sigma * A^T)_jj
        res.block(i * dim, 0, dim, dim) = denom * numer; // sigma_jj^(-1) sum(A * Sigma)_ij / sum(A * Sigma * A^T)_jj
    }
    if (normalize) {
        res.array().colwise() /= res.rowwise().sum().array();
    }
    return res;
}

//' h-step ahead Normalized Spillover
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd) {
    return fevd.bottomRows(fevd.cols()) * 100;
}

//' To-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_to_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).colwise().sum();
}

//' From-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_from_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).rowwise().sum();
}

//' Total Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
double compute_tot_spillover(Eigen::MatrixXd spillover) {
    Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
    return (spillover - diag_mat).sum() / spillover.cols();
}

//' Net Pairwise Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_net_spillover(Eigen::MatrixXd spillover) {
    return (spillover.transpose() - spillover) / spillover.cols();
}
