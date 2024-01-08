#ifndef BVHARDRAW_H
#define BVHARDRAW_H

#include "bvharomp.h"
#include "randsim.h"
#include "bvharmisc.h"

Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy);

Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat, Eigen::VectorXd DRD, Eigen::VectorXd shape, Eigen::VectorXd rate, int num_design);

Eigen::VectorXd ssvs_chol_off(Eigen::MatrixXd sse_mat, Eigen::VectorXd chol_diag, Eigen::VectorXd DRD);

Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean, Eigen::VectorXd prior_sd, Eigen::MatrixXd XtX, Eigen::VectorXd coef_ols, Eigen::MatrixXd chol_factor);

Eigen::VectorXd ssvs_dummy(Eigen::VectorXd param_obs, Eigen::VectorXd sd_numer, Eigen::VectorXd sd_denom, Eigen::VectorXd slab_weight);

Eigen::VectorXd ssvs_weight(Eigen::VectorXd param_obs, double prior_s1, double prior_s2);

Eigen::VectorXd ssvs_mn_weight(Eigen::VectorXd grp_vec, Eigen::VectorXi grp_id, Eigen::VectorXd param_obs, double prior_s1, double prior_s2);

Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec);

void varsv_regression(Eigen::Ref<Eigen::VectorXd> coef, Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec);

void varsv_ht(Eigen::Ref<Eigen::VectorXd> sv_vec, double init_sv, double sv_sig, Eigen::Ref<Eigen::VectorXd> latent_vec);

void varsv_sigh(Eigen::VectorXd& sv_sig, Eigen::VectorXd& shp, Eigen::VectorXd& scl, Eigen::VectorXd& init_sv, Eigen::MatrixXd& h1);

void varsv_h0(Eigen::VectorXd& h0, Eigen::VectorXd& prior_mean, Eigen::MatrixXd& prior_prec, Eigen::VectorXd h1, Eigen::VectorXd& sv_sig);

Eigen::MatrixXd build_shrink_mat(Eigen::VectorXd global_hyperparam, Eigen::VectorXd local_hyperparam);

Eigen::VectorXd horseshoe_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, double var, Eigen::MatrixXd shrink_mat);

Eigen::VectorXd horseshoe_fast_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat);

Eigen::VectorXd horseshoe_coef_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat);

double horseshoe_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat);

Eigen::VectorXd horseshoe_local_sparsity(Eigen::VectorXd local_latent, Eigen::VectorXd global_hyperparam, Eigen::VectorXd coef_vec, double prior_var);

double horseshoe_global_sparsity(double global_latent, Eigen::VectorXd local_hyperparam, Eigen::VectorXd coef_vec, double prior_var);

Eigen::VectorXd horseshoe_mn_global_sparsity(Eigen::VectorXd grp_vec, Eigen::VectorXi grp_id, Eigen::VectorXd global_latent, Eigen::VectorXd local_hyperparam, Eigen::VectorXd coef_vec, double prior_var);

Eigen::VectorXd horseshoe_latent(Eigen::VectorXd hyperparam);

#endif