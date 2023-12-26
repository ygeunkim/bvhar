#ifndef STRUCTURAL_H
#define STRUCTURAL_H

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month);

Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize);

Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd);

Eigen::VectorXd compute_to_spillover(Eigen::MatrixXd spillover);

Eigen::VectorXd compute_from_spillover(Eigen::MatrixXd spillover);

double compute_tot_spillover(Eigen::MatrixXd spillover);

Eigen::MatrixXd compute_net_spillover(Eigen::MatrixXd spillover);

#endif