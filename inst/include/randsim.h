#ifndef RANDSIM_H
#define RANDSIM_H

#include "bvharcommon.h"
#include "bvharprob.h"

Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method);

Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale_v);

Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape);

Rcpp::List sim_mniw(int num_sim, Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale, double shape);

Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape);

Eigen::VectorXd sim_gig(int num_sim, double lambda, double psi, double chi);

// overloading by adding rng instance
Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig, boost::random::mt19937& rng);

#endif // RANDSIM_H
