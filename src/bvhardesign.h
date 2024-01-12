#ifndef BVHARDESIGN_H
#define BVHARDESIGN_H

Eigen::MatrixXd build_y0(Eigen::MatrixXd y, int var_lag, int index);

Eigen::MatrixXd build_design(Eigen::MatrixXd y, int var_lag, bool include_mean);

Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean);

#endif