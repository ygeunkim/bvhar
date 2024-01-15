#ifndef BVHARMISC_H
#define BVHARMISC_H

typedef Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMajorMatrixXd;

Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean);

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max);

Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);

Eigen::VectorXd vectorize_eigen(Eigen::MatrixXd x);

Eigen::MatrixXd unvectorize(Eigen::VectorXd x, int num_rows, int num_cols);

Eigen::VectorXd compute_eigenvalues(Eigen::Map<Eigen::MatrixXd> x);

double mgammafn(double x, int p);

double log_mgammafn(double x, int p);

double invgamma_dens(double x, double shp, double scl, bool lg);

double compute_logml(int dim, int num_design, Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale, int posterior_shape);

Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

Eigen::MatrixXd build_cov(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

#endif
