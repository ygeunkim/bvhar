#include <mcmchs.h>

namespace bvhar {

HsParams::HsParams(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
  const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_global, const double& init_sigma,
	const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat
)
: _iter(num_iter), _x(x), _y(y),
	_init_local(init_local), _init_global(init_global), _init_sigma(init_sigma),
	_grp_id(grp_id), _grp_mat(grp_mat) {}

McmcHs::McmcHs(const HsParams& params, unsigned int seed)
: num_iter(params._iter),
	dim(params._y.cols()), dim_design(params._x.cols()), num_design(params._y.rows()),
	num_coef(dim * dim_design),
	mcmc_step(0), rng(seed),
	design_mat(kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), params._x)),
	response_vec(vectorize_eigen(params._y)),
	lambda_mat(Eigen::MatrixXd::Zero(num_coef, num_coef)),
	grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(vectorize_eigen(grp_mat)), num_grp(grp_id.size()),
	coef_draw(Eigen::VectorXd::Zero(num_coef)), sig_draw(params._init_sigma),
	local_lev(params._init_local), global_lev(params._init_global), glob_len(global_lev.size()),
	shrink_fac(Eigen::VectorXd::Zero(num_coef)),
	latent_local(Eigen::VectorXd::Zero(num_coef)),
	latent_global(Eigen::VectorXd::Zero(num_grp)),
	coef_var(Eigen::VectorXd::Zero(num_coef)),
	coef_var_loc(Eigen::MatrixXd::Zero(dim_design, dim)),
	coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
	local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
	global_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
	sig_record(Eigen::VectorXd::Zero(num_iter + 1)),
	shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)) {}

void McmcHs::addStep() {
	mcmc_step++;
}

void McmcHs::updateCoefCov() {
	for (int j = 0; j < num_grp; j++) {
		coef_var_loc = (grp_mat.array() == grp_id[j]).select(
			global_lev.segment(j, 1).replicate(dim_design, dim),
			coef_var_loc
		);
	}
	coef_var = vectorize_eigen(coef_var_loc);
	build_shrink_mat(lambda_mat, coef_var, local_lev);
	shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
}

void McmcHs::updateCoef() {
	horseshoe_coef(coef_draw, response_vec, design_mat, sig_draw, lambda_mat, rng);
	sig_draw = horseshoe_var(response_vec, design_mat, lambda_mat, rng);
}

void McmcHs::updateCov() {
	horseshoe_latent(latent_local, local_lev, rng);
	horseshoe_latent(latent_global, global_lev, rng);
	horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_draw, sig_draw, rng);
	horseshoe_mn_global_sparsity(global_lev, grp_vec, grp_id, latent_global, local_lev, coef_draw, sig_draw, rng);
}

void McmcHs::updateRecords() {
	shrink_record.row(mcmc_step) = shrink_fac;
	coef_record.row(mcmc_step) = coef_draw;
	sig_record[mcmc_step] = sig_draw;
	local_record.row(mcmc_step) = local_lev;
	global_record.row(mcmc_step) = global_lev;
}

void McmcHs::doPosteriorDraws() {
	std::lock_guard<std::mutex> lock(mtx);
	addStep();
	updateCoefCov();
	updateCoef();
	updateCov();
	updateRecords();
}

Rcpp::List McmcHs::returnRecords(int num_burn, int thin) const {
	return Rcpp::List::create(
		Rcpp::Named("alpha_record") = thin_record(coef_record, num_iter, num_burn, thin),
    Rcpp::Named("lambda_record") = thin_record(local_record, num_iter, num_burn, thin),
    Rcpp::Named("tau_record") = thin_record(global_record, num_iter, num_burn, thin),
    Rcpp::Named("sigma_record") = thin_vec_record(sig_record, num_iter, num_burn, thin),
    Rcpp::Named("kappa_record") = thin_record(shrink_record, num_iter, num_burn, thin)
	);
}

BlockHs::BlockHs(const HsParams& params, unsigned int seed)
: McmcHs(params, seed),
	block_coef(Eigen::VectorXd::Zero(num_coef + 1)) {}

void BlockHs::updateCoef() {
	horseshoe_coef_var(block_coef, response_vec, design_mat, lambda_mat, rng);
}

void BlockHs::updateRecords() {
	shrink_record.row(mcmc_step) = shrink_fac;
	coef_record.row(mcmc_step) = block_coef.tail(num_coef);
	sig_record[mcmc_step] = block_coef[0];
	local_record.row(mcmc_step) = local_lev;
	global_record.row(mcmc_step) = global_lev;
}

FastHs::FastHs(const HsParams& params, unsigned int seed) : McmcHs(params, seed) {}

void FastHs::updateCoef() {
	horseshoe_fast_coef(
		coef_draw,
		response_vec / sqrt(sig_draw),
		design_mat / sqrt(sig_draw),
		sig_draw * lambda_mat,
		rng
	);
	sig_draw = horseshoe_var(response_vec, design_mat, lambda_mat, rng);
}

void FastHs::updateRecords() {
	shrink_record.row(mcmc_step) = shrink_fac;
	coef_record.row(mcmc_step) = coef_draw;
	sig_record[mcmc_step] = sig_draw;
	local_record.row(mcmc_step) = local_lev;
	global_record.row(mcmc_step) = global_lev;
}

} // namespace bvhar