#ifndef BVHARMISC_H
#define BVHARMISC_H

Eigen::MatrixXd scale_har(int dim, int week, int month);

Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max);

Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::Map<Eigen::MatrixXd> mat_scale_u, Eigen::Map<Eigen::MatrixXd> mat_scale_v);

Rcpp::List sim_mniw(int num_sim, Eigen::MatrixXd mat_mean, Eigen::Map<Eigen::MatrixXd> mat_scale_u, Eigen::Map<Eigen::MatrixXd> mat_scale, double shape);

Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);

Eigen::VectorXd vectorize_eigen(Eigen::MatrixXd x);

Eigen::VectorXd compute_eigenvalues(Eigen::Map<Eigen::MatrixXd> x);

double mgammafn(double x, int p);

double log_mgammafn(double x, int p);

#endif
