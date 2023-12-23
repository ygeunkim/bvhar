#ifndef FITVAR_H
#define FITVAR_H

Eigen::MatrixXd forecast_var(Rcpp::List object, int step);

Eigen::MatrixXd forecast_vhar(Rcpp::List object, int step);

Rcpp::List forecast_bvar(Rcpp::List object, int step, int num_sim);

Rcpp::List forecast_bvharmn(Rcpp::List object, int step, int num_sim);

Eigen::MatrixXd forecast_bvarsv(int var_lag, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd coef_mat);

Eigen::MatrixXd forecast_bvharsv(int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd coef_mat, Eigen::MatrixXd HARtrans);

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month);

#endif
