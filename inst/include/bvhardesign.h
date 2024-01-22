#ifndef BVHARDESIGN_H
#define BVHARDESIGN_H

#include <RcppEigen.h>

Eigen::MatrixXd build_y0(Eigen::MatrixXd y, int var_lag, int index);

Eigen::MatrixXd build_design(Eigen::MatrixXd y, int var_lag, bool include_mean);

Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean);

Eigen::MatrixXd build_ydummy(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly, bool include_mean);

Eigen::MatrixXd build_xdummy(Eigen::VectorXd lag_seq, double lambda, Eigen::VectorXd sigma, double eps, bool include_mean);

#endif