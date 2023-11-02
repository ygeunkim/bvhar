#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' h-step ahead Forecast Error Variance Decomposition
//' 
//' [w_(h = 1, ij)^T, w_(h = 2, ij)^T, ...]
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat) {
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
        ma_prod = cov_mat * vma_coef.block(i * dim, 0, dim, dim);
        innov_account += vma_coef.block(i * dim, 0, dim, dim).transpose() * ma_prod;
        numer.array() += (cov_diag * ma_prod).array().square();
        denom.diagonal() = 1 / innov_account.diagonal().array();
        // res.block(i * dim, 0, dim, dim) = (numer * denom).transpose();
        res.block(i * dim, 0, dim, dim) = numer * denom;
    }
    return res;
}

//' h-step ahead Normalized Spillover
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd) {
    int dim = fevd.cols();
    int step = fevd.rows() / dim;
    Eigen::MatrixXd block_mat(dim, dim);
    Eigen::VectorXd col_sum(dim);
    for (int i = 0; i < step; i++) {
        block_mat = fevd.block(i * dim, 0, dim, dim);
        col_sum = block_mat.colwise().sum();
        block_mat.array().colwise() /= col_sum.array();
        fevd.block(i * dim, 0, dim, dim) = block_mat;
        // Eigen::VectorXd tmp = fevd.rowwise().sum().eval();
        // return fevd.array().rowwise() / tmp.array();
        // fevd = fevd.rowwise().normalized();
    }
    return fevd.bottomRows(dim) * 100;
}

//' h-step ahead Spillover Index
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_sp_index(Eigen::MatrixXd spillover) {
    int dim = spillover.cols();
    Eigen::MatrixXd diag_mat = Eigen::MatrixXd::Zero(dim, dim);
    diag_mat.diagonal() = spillover.diagonal();
    return (spillover - diag_mat).colwise().sum();
}
