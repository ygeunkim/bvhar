#include "mcmcsv.h"

McmcSv::McmcSv(
	const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
	const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec
)
: x(x), y(y),
	prior_sig_shp(prior_sig_shp), prior_sig_scl(prior_sig_scl),
	prior_init_mean(prior_init_mean), prior_init_prec(prior_init_prec),
	dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
	chol_lower(Eigen::MatrixXd::Zero(dim, dim)),
	prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
	prior_alpha_prec(Eigen::MatrixXd::Zero(num_coef, num_coef)),
	prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
	prior_chol_prec(Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol)),
	ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
	prior_mean_j(Eigen::VectorXd::Zero(dim_design)),
	prior_prec_j(Eigen::MatrixXd::Identity(dim_design, dim_design)),
	response_contem(Eigen::VectorXd::Zero(num_design)),
	sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
	contem_id(0) {
  num_lowerchol = dim * (dim - 1) / 2;
  num_coef = dim * dim_design;
	coef_mat = (x.transpose() * x).llt().solve(x.transpose() * y);
	coef_vec = vectorize_eigen(coef_vec);
	contem_coef = Eigen::VectorXd(num_lowerchol);
	latent_innov = y - x * coef_mat;
	lvol_init = latent_innov.transpose().array().square().rowwise().mean().log();
	lvol_draw = lvol_init.replicate(num_design, 1);
	lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
  coef_j = coef_mat;
  coef_j.col(0) = Eigen::VectorXd::Zero(dim_design);
}

void McmcSv::UpdateCoef() {
	chol_lower = build_inv_lower(dim, contem_coef);
	sqrt_sv = (-lvol_draw / 2).array().exp();
	for (int j = 0; j < dim; j++) {
		prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
		prior_prec_j = prior_alpha_prec.block(dim_design * j, dim_design * j, dim_design, dim_design);
		coef_j = coef_mat;
		Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
		Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
		Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * vectorize_eigen(sqrt_sv_j).array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
		Eigen::VectorXd response_j = vectorize_eigen(
			((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array() // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
    ); // Response vector of j-th column coef equation: n(k - j + 1)-dim
		coef_mat.col(j) = varsv_regression(
			design_coef, response_j,
      prior_mean_j, prior_prec_j
    );
	}
	coef_vec = vectorize_eigen(coef_mat);
}

void McmcSv::UpdateState() {
	latent_innov = y - x * coef_mat;
  ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
	ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
	for (int t = 0; t < dim; t++) {
		lvol_draw.col(t) = varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t));
	}
}

void McmcSv::UpdateImpact() {
	for (int j = 2; j < dim + 1; j++) {
		response_contem = latent_innov.col(j - 2).array() * sqrt_sv.col(j - 2).array(); // n-dim
		Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * vectorize_eigen(sqrt_sv.col(j - 2)).array(); // n x (j - 1)
		contem_id = (j - 1) * (j - 2) / 2;
		contem_coef = varsv_regression(
			design_contem, response_contem,
			prior_chol_mean.segment(contem_id, j - 1),
			prior_chol_prec.block(contem_id, contem_id, j - 1, j - 1)
		);
	}
}

void McmcSv::UpdateStateVar() {
	lvol_sig = varsv_sigh(prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw);
}

void McmcSv::UpdateInitState() {
	lvol_init = varsv_h0(prior_init_mean, prior_init_prec, lvol_draw.row(0), lvol_sig);
}

SsvsSv::SsvsSv(
	const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
		const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
		const Eigen::VectorXd& coef_spike, const Eigen::VectorXd& coef_slab,
    const Eigen::VectorXd& coef_slab_weight, const Eigen::VectorXd& chol_spike,
    const Eigen::VectorXd& chol_slab, const Eigen::VectorXd& chol_slab_weight,
    const double& coef_s1, const double& coef_s2,
    const double& chol_s1, const double& chol_s2,
    const Eigen::VectorXd& mean_non, const double& sd_non, const bool& include_mean
)
: McmcSv(x, y, prior_sig_shp, prior_sig_scl, prior_init_mean, prior_init_prec),
	include_mean(include_mean),
	coef_weight(coef_slab_weight),
	contem_weight(chol_slab_weight),
	contem_dummy(Eigen::VectorXd::Ones(num_lowerchol)),
	num_grp(grp_id.size()),
	grp_id(grp_id),
	grp_mat(grp_mat),
	grp_vec(vectorize_eigen(grp_mat)),
	coef_s1(coef_s1),
	coef_s2(coef_s2),
	contem_s1(contem_s1),
	contem_s2(contem_s2),
	prior_sd_non(sd_non),
	prior_sd(Eigen::VectorXd(num_coef)) {
	num_alpha = num_coef - dim;
	if (!include_mean) {
		num_alpha += dim; // always dim^2 p
	}
	if (include_mean) {
    for (int j = 0; j < dim; j++) {
      prior_alpha_mean.segment(j * dim_design, num_alpha / dim) = Eigen::VectorXd::Zero(num_alpha / dim);
      prior_alpha_mean[j * dim_design + num_alpha / dim] = mean_non[j];
    }
  }
	coef_dummy = Eigen::VectorXd::Ones(num_alpha);
	slab_weight = Eigen::VectorXd::Zero(num_alpha);
	slab_weight_mat = Eigen::MatrixXd::Zero(num_alpha / dim, dim);
	coef_mixture_mat = Eigen::VectorXd::Zero(num_alpha);
}

HorseshoeSv::HorseshoeSv(
	const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
	const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec,
	const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
	const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_global,
	const Eigen::VectorXd& init_contem_local, const Eigen::VectorXd& init_contem_global
)
: McmcSv(x, y, prior_sig_shp, prior_sig_scl, prior_init_mean, prior_init_prec),
	local_lev(init_local),
	global_lev(init_global),
	shrink_fac(Eigen::VectorXd::Zero(num_coef)),
	num_grp(grp_id.size()),
	grp_id(grp_id),
	grp_mat(grp_mat),
	grp_vec(vectorize_eigen(grp_mat)),
	latent_local(Eigen::VectorXd::Zero(num_coef)),
	latent_global(Eigen::VectorXd::Zero(1)),
	coef_var(Eigen::VectorXd::Zero(num_coef)),
	coef_var_loc(Eigen::MatrixXd::Zero(num_design, dim)),
	contem_local_lev(init_contem_local),
	contem_global_lev(init_contem_global),
	contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
	latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)),
	latent_contem_global(Eigen::VectorXd::Zero(1)) {}

void HorseshoeSv::UpdateCoefPrec() {
	for (int j = 0; j < num_grp; j++) {
		coef_var_loc = (grp_mat.array() == grp_id[j]).select(
			global_lev.segment(j, 1).replicate(dim_design, dim),
			coef_var_loc
		);
	}
	coef_var = vectorize_eigen(coef_var_loc);
	prior_alpha_prec = build_shrink_mat(coef_var, local_lev);
	shrink_fac = 1 / (1 + prior_alpha_prec.diagonal().array());
}

void HorseshoeSv::UpdateCoefShrink() {
	latent_local = horseshoe_latent(local_lev);
	latent_global = horseshoe_latent(global_lev);
	local_lev = horseshoe_local_sparsity(latent_local, coef_var, coef_vec, 1);
	global_lev = horseshoe_mn_global_sparsity(grp_vec, grp_id, latent_global, local_lev, coef_vec, 1);
}

void HorseshoeSv::UpdateImpactPrec() {
	latent_contem_local = horseshoe_latent(contem_local_lev);
	latent_contem_global = horseshoe_latent(contem_global_lev);
	contem_var = vectorize_eigen(contem_global_lev.replicate(1, num_lowerchol));
	contem_global_lev = horseshoe_local_sparsity(latent_contem_local, contem_var, contem_coef, 1);
	prior_chol_prec = build_shrink_mat(contem_var, contem_global_lev);
}