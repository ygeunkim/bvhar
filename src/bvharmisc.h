#ifndef BVHARMISC_H
#define BVHARMISC_H

Eigen::MatrixXd scale_har(int m);

Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max);

Rcpp::List sim_mniw(int num_sim, Eigen::MatrixXd mat_mean, Eigen::Map<Eigen::MatrixXd> mat_scale_u, Eigen::Map<Eigen::MatrixXd> mat_scale, double shape);

#endif
