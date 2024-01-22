#ifndef BVHARDRAW_H
#define BVHARDRAW_H

#include "randsim.h"

double mgammafn(double x, int p);

double log_mgammafn(double x, int p);

double invgamma_dens(double x, double shp, double scl, bool lg);

double compute_logml(int dim, int num_design, Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale, int posterior_shape);

Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy);

void ssvs_chol_diag(Eigen::VectorXd& chol_diag, Eigen::MatrixXd& sse_mat, Eigen::VectorXd& DRD, Eigen::VectorXd& shape, Eigen::VectorXd& rate, int num_design, boost::mt19937& rng);

void ssvs_chol_off(Eigen::VectorXd& chol_off, Eigen::MatrixXd& sse_mat, Eigen::VectorXd& chol_diag, Eigen::VectorXd& DRD, boost::random::mt19937& rng);

Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

Eigen::MatrixXd build_cov(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

void ssvs_coef(Eigen::VectorXd& coef, Eigen::VectorXd& prior_mean, Eigen::VectorXd& prior_sd, Eigen::MatrixXd& XtX, Eigen::VectorXd& coef_ols, Eigen::MatrixXd& chol_factor, boost::random::mt19937& rng);

void ssvs_dummy(Eigen::VectorXd& dummy, Eigen::VectorXd param_obs, Eigen::VectorXd& sd_numer, Eigen::VectorXd& sd_denom, Eigen::VectorXd& slab_weight, boost::random::mt19937& rng);

void ssvs_weight(Eigen::VectorXd& weight, Eigen::VectorXd param_obs, double prior_s1, double prior_s2, boost::random::mt19937& rng);

void ssvs_mn_weight(Eigen::VectorXd& weight, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id, Eigen::VectorXd& param_obs, double prior_s1, double prior_s2, boost::random::mt19937& rng);

Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec);

void varsv_regression(Eigen::Ref<Eigen::VectorXd> coef, Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec, boost::random::mt19937& rng);

void varsv_ht(Eigen::Ref<Eigen::VectorXd> sv_vec, double init_sv, double sv_sig, Eigen::Ref<Eigen::VectorXd> latent_vec, boost::random::mt19937& rng);

void varsv_sigh(Eigen::VectorXd& sv_sig, Eigen::VectorXd& shp, Eigen::VectorXd& scl, Eigen::VectorXd& init_sv, Eigen::MatrixXd& h1, boost::random::mt19937& rng);

void varsv_h0(Eigen::VectorXd& h0, Eigen::VectorXd& prior_mean, Eigen::MatrixXd& prior_prec, Eigen::VectorXd h1, Eigen::VectorXd& sv_sig, boost::random::mt19937& rng);

void build_shrink_mat(Eigen::MatrixXd& cov, Eigen::VectorXd& global_hyperparam, Eigen::VectorXd& local_hyperparam);

void horseshoe_coef(Eigen::VectorXd& coef, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, double var, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng);

void horseshoe_fast_coef(Eigen::VectorXd& coef, Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat, boost::random::mt19937& rng);

void horseshoe_coef_var(Eigen::VectorXd& coef_var, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng);

double horseshoe_var(Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng);

void horseshoe_local_sparsity(Eigen::VectorXd& local_lev, Eigen::VectorXd& local_latent, Eigen::VectorXd& global_hyperparam, Eigen::VectorXd coef_vec, double prior_var, boost::random::mt19937& rng);

double horseshoe_global_sparsity(double global_latent, Eigen::VectorXd& local_hyperparam, Eigen::VectorXd& coef_vec, double prior_var, boost::random::mt19937& rng);

void horseshoe_mn_global_sparsity(Eigen::VectorXd& global_lev, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id, Eigen::VectorXd& global_latent, Eigen::VectorXd& local_hyperparam, Eigen::VectorXd coef_vec, double prior_var, boost::random::mt19937& rng);

void horseshoe_latent(Eigen::VectorXd& latent, Eigen::VectorXd& hyperparam, boost::random::mt19937& rng);

Eigen::MatrixXd thin_record(const Eigen::MatrixXd& record, int num_iter, int num_burn, int thin);

Eigen::VectorXd thin_vec_record(const Eigen::VectorXd& record, int num_iter, int num_burn, int thin);

#endif