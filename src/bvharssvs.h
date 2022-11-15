#ifndef BVHARSSVS_H
#define BVHARSSVS_H

Eigen::MatrixXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy);

Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat, Eigen::MatrixXd inv_DRD, Eigen::VectorXd shape, Eigen::VectorXd rate, int num_design);

Eigen::VectorXd ssvs_chol_off(Eigen::MatrixXd sse_mat, Eigen::VectorXd chol_diag, Eigen::MatrixXd inv_DRD);

Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

Eigen::VectorXd ssvs_chol_dummy(Eigen::VectorXd chol_upper, Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd slab_weight);

Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec, Eigen::MatrixXd XtX, Eigen::VectorXd coef_ols, Eigen::MatrixXd chol_factor);

Eigen::VectorXd ssvs_coef_dummy(Eigen::VectorXd coef, Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd slab_weight);

#endif
