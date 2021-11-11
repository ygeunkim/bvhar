#ifndef FITVAR_H
#define FITVAR_H

Eigen::MatrixXd forecast_var(Rcpp::List object, int step);

Eigen::MatrixXd forecast_vhar(Rcpp::List object, int step);

Rcpp::List forecast_bvar(Rcpp::List object, int step, int num_sim);

Rcpp::List forecast_bvharmn(Rcpp::List object, int step, int num_sim);

#endif
