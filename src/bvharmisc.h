#ifndef BVHARMISC_H
#define BVHARMISC_H

SEXP build_y0(Eigen::MatrixXd x, int p, int t);

SEXP build_design(Eigen::MatrixXd x, int p);

SEXP diag_misc(Eigen::VectorXd x);

SEXP build_ydummy(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd delta);

SEXP build_xdummy(int p, double lambda, Eigen::VectorXd sigma, double eps);

SEXP minnesota_prior (Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy);

SEXP build_ydummy_bvhar(Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly);

SEXP VARtoVMA(Rcpp::List object, int lag_max);

SEXP compute_covmse(Rcpp::List object, int step);

#endif
