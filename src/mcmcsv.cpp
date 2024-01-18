#include "mcmcsv.h"

SvParams::SvParams(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	Rcpp::List& spec
)
: _iter(num_iter), _x(x), _y(y),
	_sig_shp(Rcpp::as<Eigen::VectorXd>(spec["shape"])),
	_sig_scl(Rcpp::as<Eigen::VectorXd>(spec["scale"])),
	_init_mean(Rcpp::as<Eigen::VectorXd>(spec["initial_mean"])),
	_init_prec(Rcpp::as<Eigen::MatrixXd>(spec["initial_prec"])) {}

MinnParams::MinnParams(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	Rcpp::List& sv_spec, Rcpp::List& priors
)
: SvParams(num_iter, x, y, sv_spec),
	_prior_mean(Rcpp::as<Eigen::MatrixXd>(priors["prior_mean"])),
	_prior_prec(Rcpp::as<Eigen::MatrixXd>(priors["prior_prec"])),
	_prec_diag(Rcpp::as<Eigen::MatrixXd>(priors["sigma"])) {}

SsvsParams::SsvsParams(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	Rcpp::List& sv_spec,
	const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
	Rcpp::List& ssvs_spec,
	bool include_mean
)
: SvParams(num_iter, x, y, sv_spec),
	_grp_id(grp_id), _grp_mat(grp_mat),
	_coef_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_spike"])),
	_coef_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_slab"])),
	_coef_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_mixture"])),
	_contem_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_spike"])),
	_contem_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_slab"])),
	_contem_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_mixture"])),
	_coef_s1(ssvs_spec["coef_s1"]), _coef_s2(ssvs_spec["coef_s2"]),
	_contem_s1(ssvs_spec["chol_s1"]), _contem_s2(ssvs_spec["chol_s2"]),
	_mean_non(Rcpp::as<Eigen::VectorXd>(ssvs_spec["mean_non"])),
	_sd_non(ssvs_spec["sd_non"]), _mean(include_mean) {}

HorseshoeParams::HorseshoeParams(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	Rcpp::List& sv_spec,
	const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat
)
: SvParams(num_iter, x, y, sv_spec),
	_grp_id(grp_id), _grp_mat(grp_mat) {}
	// _init_local(Rcpp::as<Eigen::VectorXd>(hs_spec["local_sparsity"])),
	// _init_global(Rcpp::as<Eigen::VectorXd>(hs_spec["global_sparsity"])),
	// _init_contem_local(Rcpp::as<Eigen::VectorXd>(hs_spec["contem_local_sparsity"])),
	// _init_conetm_global(Rcpp::as<Eigen::VectorXd>(hs_spec["contem_global_sparsity"])) {}

SvInits::SvInits(const SvParams& params) {
	_coef = (params._x.transpose() * params._x).llt().solve(params._x.transpose() * params._y); // OLS
	int dim = params._y.cols();
	int num_lowerchol = dim * (dim - 1) / 2;
	int num_design = params._y.rows();
	_contem = .001 * Eigen::VectorXd::Zero(num_lowerchol);
	_lvol_init = (params._y - params._x * _coef).transpose().array().square().rowwise().mean().log();
	_lvol = _lvol_init.transpose().replicate(num_design, 1);
	_lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
}

SvInits::SvInits(Rcpp::List& init)
: _coef(Rcpp::as<Eigen::MatrixXd>(init["init_coef"])),
	_contem(Rcpp::as<Eigen::VectorXd>(init["init_contem"])),
	_lvol_init(Rcpp::as<Eigen::VectorXd>(init["lvol_init"])),
	_lvol(Rcpp::as<Eigen::MatrixXd>(init["lvol"])),
	_lvol_sig(Rcpp::as<Eigen::VectorXd>(init["lvol_sig"])) {}

SsvsInits::SsvsInits(Rcpp::List& init)
: SvInits(init),
	_coef_dummy(Rcpp::as<Eigen::VectorXd>(init["init_coef_dummy"])),
	_coef_weight(Rcpp::as<Eigen::VectorXd>(init["coef_mixture"])),
	_contem_weight(Rcpp::as<Eigen::VectorXd>(init["chol_mixture"])) {}

HorseshoeInits::HorseshoeInits(Rcpp::List& init)
: SvInits(init),
	_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
	_init_global(Rcpp::as<Eigen::VectorXd>(init["global_sparsity"])),
	_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
	_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}

McmcSv::McmcSv(const SvParams& params, const SvInits& inits, unsigned int seed)
: x(params._x), y(params._y),
	prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl),
	prior_init_mean(params._init_mean), prior_init_prec(params._init_prec),
	num_iter(params._iter),
	coef_mat(inits._coef), contem_coef(inits._contem),
	lvol_init(inits._lvol_init), lvol_draw(inits._lvol), lvol_sig(inits._lvol_sig),
	dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
	ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
	prior_mean_j(Eigen::VectorXd::Zero(dim_design)),
	prior_prec_j(Eigen::MatrixXd::Identity(dim_design, dim_design)),
	response_contem(Eigen::VectorXd::Zero(num_design)),
	sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
	contem_id(0),
	mcmc_step(0),
	rng(seed) {
  num_lowerchol = dim * (dim - 1) / 2;
  num_coef = dim * dim_design;
	prior_alpha_mean = Eigen::VectorXd::Zero(num_coef);
	prior_alpha_prec = Eigen::MatrixXd::Zero(num_coef, num_coef);
	prior_chol_mean = Eigen::VectorXd::Zero(num_lowerchol);
	prior_chol_prec = Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol);
	coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef);
	contem_coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol);
	lvol_sig_record = Eigen::MatrixXd::Ones(num_iter + 1, dim);
	lvol_init_record = Eigen::MatrixXd::Zero(num_iter + 1, dim);
	lvol_record = Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim);
	// coef_mat = (x.transpose() * x).llt().solve(x.transpose() * y);
	coef_vec = vectorize_eigen(coef_mat);
	// contem_coef = .001 * Eigen::VectorXd::Zero(num_lowerchol);
	latent_innov = y - x * coef_mat;
	chol_lower = build_inv_lower(dim, contem_coef);
	// lvol_init = latent_innov.transpose().array().square().rowwise().mean().log();
	lvol_init_record.row(0) = lvol_init;
	// lvol_draw = lvol_init.transpose().replicate(num_design, 1);
	// lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
	coef_record.row(0) = vectorize_eigen(coef_mat);
	lvol_init_record.row(0) = lvol_init;
	lvol_record.row(0) = vectorize_eigen(lvol_draw.transpose());
  coef_j = coef_mat;
}

void McmcSv::updateCoef() {
	for (int j = 0; j < dim; j++) {
		prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
		prior_prec_j = prior_alpha_prec.block(dim_design * j, dim_design * j, dim_design, dim_design);
		coef_j = coef_mat;
		coef_j.col(j).setZero();
		Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
		Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
		Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * vectorize_eigen(sqrt_sv_j).array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
		Eigen::VectorXd response_j = vectorize_eigen(
			((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array() // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
    ); // Response vector of j-th column coef equation: n(k - j + 1)-dim
		varsv_regression(
			coef_mat.col(j),
			design_coef, response_j,
      prior_mean_j, prior_prec_j,
			rng
    );
	}
	coef_vec = vectorize_eigen(coef_mat);
}

void McmcSv::updateState() {
  ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
	ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
	for (int t = 0; t < dim; t++) {
		varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t), rng);
	}
}

void McmcSv::updateImpact() {
	for (int j = 2; j < dim + 1; j++) {
		response_contem = latent_innov.col(j - 2).array() * sqrt_sv.col(j - 2).array(); // n-dim
		Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * vectorize_eigen(sqrt_sv.col(j - 2)).array(); // n x (j - 1)
		contem_id = (j - 1) * (j - 2) / 2;
		varsv_regression(
			contem_coef.segment(contem_id, j - 1),
			design_contem, response_contem,
			prior_chol_mean.segment(contem_id, j - 1),
			prior_chol_prec.block(contem_id, contem_id, j - 1, j - 1),
			rng
		);
	}
}

void McmcSv::updateStateVar() {
	varsv_sigh(lvol_sig, prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw, rng);
}

void McmcSv::updateInitState() {
	varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), lvol_sig, rng);
}

void McmcSv::addStep() {
	mcmc_step++;
}

MinnSv::MinnSv(const MinnParams& params, const SvInits& inits, unsigned int seed)
: McmcSv(params, inits, seed) {
	prior_alpha_mean = vectorize_eigen(params._prior_mean);
	prior_alpha_prec = kronecker_eigen(params._prec_diag, params._prior_prec);
}

void MinnSv::updateRecords() {
	std::lock_guard<std::mutex> lock(mtx);
	coef_record.row(mcmc_step) = coef_vec;
	contem_coef_record.row(mcmc_step) = contem_coef;
	lvol_record.row(mcmc_step) = vectorize_eigen(lvol_draw.transpose());
	lvol_sig_record.row(mcmc_step) = lvol_sig;
	lvol_init_record.row(mcmc_step) = lvol_init;
}

void MinnSv::doPosteriorDraws() {
	sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
	updateCoef();
	latent_innov = y - x * coef_mat; // E_t before a
	updateImpact();
	chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
	updateState();
	updateStateVar();
	updateInitState();
	updateRecords();
}

Rcpp::List MinnSv::returnRecords(int num_burn, int thin) const {
	Rcpp::List res = Rcpp::List::create(
		Rcpp::Named("alpha_record") = coef_record,
    Rcpp::Named("h_record") = lvol_record,
    Rcpp::Named("a_record") = contem_coef_record,
    Rcpp::Named("h0_record") = lvol_init_record,
    Rcpp::Named("sigh_record") = lvol_sig_record
  );
	for (auto& record : res) {
		record = thin_record(record, num_iter, num_burn, thin);
	}
	return res;
}

SsvsSv::SsvsSv(const SsvsParams& params, const SsvsInits& inits, unsigned int seed)
: McmcSv(params, inits, seed),
	include_mean(params._mean),
	grp_id(params._grp_id),
	num_grp(grp_id.size()),
	grp_mat(params._grp_mat),
	grp_vec(vectorize_eigen(grp_mat)),
	// coef_weight(params._coef_weight),
	// contem_weight(params._contem_weight),
	coef_weight(inits._coef_weight),
	contem_weight(inits._contem_weight),
	coef_dummy(inits._coef_dummy),
	contem_dummy(Eigen::VectorXd::Ones(num_lowerchol)),
	coef_spike(params._coef_spike),
	coef_slab(params._coef_slab),
	contem_spike(params._contem_spike),
	contem_slab(params._contem_slab),
	coef_s1(params._coef_s1),
	coef_s2(params._coef_s2),
	contem_s1(params._contem_s1),
	contem_s2(params._contem_s2),
	prior_sd_non(params._sd_non),
	prior_sd(Eigen::VectorXd(num_coef)) {
	num_alpha = num_coef - dim;
	if (!include_mean) {
		num_alpha += dim; // always dim^2 p
	}
	if (include_mean) {
    for (int j = 0; j < dim; j++) {
      prior_alpha_mean.segment(j * dim_design, num_alpha / dim) = Eigen::VectorXd::Zero(num_alpha / dim);
      prior_alpha_mean[j * dim_design + num_alpha / dim] = params._mean_non[j];
    }
  }
	coef_dummy_record = Eigen::MatrixXd::Ones(num_iter + 1, num_alpha);
	coef_weight_record = Eigen::MatrixXd::Zero(num_iter + 1, num_grp);
	contem_dummy_record = Eigen::MatrixXd::Ones(num_iter + 1, num_lowerchol);
	contem_weight_record = Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol);
	coef_weight_record.row(0) = coef_weight;
	contem_weight_record.row(0) = contem_weight;
	// coef_dummy = Eigen::VectorXd::Ones(num_alpha);
	slab_weight = Eigen::VectorXd::Ones(num_alpha);
	slab_weight_mat = Eigen::MatrixXd::Ones(num_alpha / dim, dim);
	coef_mixture_mat = Eigen::VectorXd::Zero(num_alpha);
}

void SsvsSv::updateCoefPrec() {
	coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy);
	if (include_mean) {
		for (int j = 0; j < dim; j++) {
			prior_sd.segment(j * dim_design, num_alpha / dim) = coef_mixture_mat.segment(
				j * num_alpha / dim,
				num_alpha / dim
			);
			prior_sd[j * dim_design + num_alpha / dim] = prior_sd_non;
		}
	} else {
		prior_sd = coef_mixture_mat;
	}
	prior_alpha_prec.setZero();
	prior_alpha_prec.diagonal() = 1 / prior_sd.array().square();
}

void SsvsSv::updateCoefShrink() {
	for (int j = 0; j < num_grp; j++) {
		slab_weight_mat = (grp_mat.array() == grp_id[j]).select(
			coef_weight.segment(j, 1).replicate(num_alpha / dim, dim),
			slab_weight_mat
		);
	}
	slab_weight = vectorize_eigen(slab_weight_mat);
	ssvs_dummy(
		coef_dummy,
		vectorize_eigen(coef_mat.topRows(num_alpha / dim)),
		coef_slab, coef_spike, slab_weight,
		rng
	);
	ssvs_mn_weight(coef_weight, grp_vec, grp_id, coef_dummy, coef_s1, coef_s2, rng);
	// coef_weight_record.row(mcmc_step) = coef_weight;
}

void SsvsSv::updateImpactPrec() {
	ssvs_dummy(contem_dummy, contem_coef, contem_slab, contem_spike, contem_weight, rng);
	ssvs_weight(contem_weight, contem_dummy, contem_s1, contem_s2, rng);
	prior_chol_prec.diagonal() = 1 / build_ssvs_sd(contem_spike, contem_slab, contem_dummy).array().square();
	// contem_dummy_record.row(mcmc_step) = contem_dummy;
	// contem_weight_record.row(mcmc_step) = contem_weight;
}

void SsvsSv::updateRecords() {
	std::lock_guard<std::mutex> lock(mtx);
	coef_record.row(mcmc_step) = coef_vec;
	contem_coef_record.row(mcmc_step) = contem_coef;
	lvol_record.row(mcmc_step) = vectorize_eigen(lvol_draw.transpose());
	lvol_sig_record.row(mcmc_step) = lvol_sig;
	lvol_init_record.row(mcmc_step) = lvol_init;
	coef_weight_record.row(mcmc_step) = coef_weight;
	contem_dummy_record.row(mcmc_step) = contem_dummy;
	contem_weight_record.row(mcmc_step) = contem_weight;
}

void SsvsSv::doPosteriorDraws() {
	updateCoefPrec();
	sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
	updateCoef();
	updateCoefShrink();
	updateImpactPrec();
	latent_innov = y - x * coef_mat; // E_t before a
	updateImpact();
	chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
	updateState();
	updateStateVar();
	updateInitState();
	updateRecords();
}

Rcpp::List SsvsSv::returnRecords(int num_burn, int thin) const {
	Rcpp::List res = Rcpp::List::create(
		Rcpp::Named("alpha_record") = coef_record,
    Rcpp::Named("h_record") = lvol_record,
    Rcpp::Named("a_record") = contem_coef_record,
    Rcpp::Named("h0_record") = lvol_init_record,
    Rcpp::Named("sigh_record") = lvol_sig_record,
    Rcpp::Named("gamma_record") = coef_dummy_record
  );
	for (auto& record : res) {
		record = thin_record(record, num_iter, num_burn, thin);
	}
	return res;
}

HorseshoeSv::HorseshoeSv(const HorseshoeParams& params, const HorseshoeInits& inits, unsigned int seed)
: McmcSv(params, inits, seed),
	grp_id(params._grp_id),
	num_grp(grp_id.size()),
	grp_mat(params._grp_mat),
	grp_vec(vectorize_eigen(grp_mat)),
	// local_lev(params._init_local),
	// global_lev(params._init_global),
	local_lev(inits._init_local),
	global_lev(inits._init_global),
	shrink_fac(Eigen::VectorXd::Zero(num_coef)),
	latent_local(Eigen::VectorXd::Zero(num_coef)),
	latent_global(Eigen::VectorXd::Zero(num_grp)),
	coef_var(Eigen::VectorXd::Zero(num_coef)),
	coef_var_loc(Eigen::MatrixXd::Zero(dim_design, dim)),
	// contem_local_lev(params._init_contem_local),
	// contem_global_lev(params._init_conetm_global),
	contem_local_lev(inits._init_contem_local),
	contem_global_lev(inits._init_conetm_global),
	contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
	latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)),
	latent_contem_global(Eigen::VectorXd::Zero(1)) {
	local_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef);
	global_record = Eigen::MatrixXd::Zero(num_iter + 1, num_grp);
	shrink_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef);
	local_record.row(0) = local_lev;
	global_record.row(0) = global_lev;
}

void HorseshoeSv::updateCoefPrec() {
	for (int j = 0; j < num_grp; j++) {
		coef_var_loc = (grp_mat.array() == grp_id[j]).select(
			global_lev.segment(j, 1).replicate(dim_design, dim),
			coef_var_loc
		);
	}
	coef_var = vectorize_eigen(coef_var_loc);
	build_shrink_mat(prior_alpha_prec, coef_var, local_lev);
	shrink_fac = 1 / (1 + prior_alpha_prec.diagonal().array());
	// shrink_record.row(mcmc_step) = shrink_fac;
}

void HorseshoeSv::updateCoefShrink() {
	horseshoe_latent(latent_local, local_lev, rng);
	horseshoe_latent(latent_global, global_lev, rng);
	horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec, 1, rng);
	horseshoe_mn_global_sparsity(global_lev, grp_vec, grp_id, latent_global, local_lev, coef_vec, 1, rng);
	// local_record.row(mcmc_step) = local_lev;
	// global_record.row(mcmc_step) = global_lev;
}

void HorseshoeSv::updateImpactPrec() {
	horseshoe_latent(latent_contem_local, contem_local_lev, rng);
	horseshoe_latent(latent_contem_global, contem_global_lev, rng);
	contem_var = vectorize_eigen(contem_global_lev.replicate(1, num_lowerchol));
	horseshoe_local_sparsity(contem_local_lev, latent_contem_local, contem_var, contem_coef, 1, rng);
	contem_global_lev[0] = horseshoe_global_sparsity(latent_contem_global[0], latent_contem_local, contem_coef, 1, rng);
	build_shrink_mat(prior_chol_prec, contem_var, contem_local_lev);
}

void HorseshoeSv::updateRecords() {
	std::lock_guard<std::mutex> lock(mtx);
	coef_record.row(mcmc_step) = coef_vec;
	contem_coef_record.row(mcmc_step) = contem_coef;
	lvol_record.row(mcmc_step) = vectorize_eigen(lvol_draw.transpose());
	lvol_sig_record.row(mcmc_step) = lvol_sig;
	lvol_init_record.row(mcmc_step) = lvol_init;
	shrink_record.row(mcmc_step) = shrink_fac;
	local_record.row(mcmc_step) = local_lev;
	global_record.row(mcmc_step) = global_lev;
}

void HorseshoeSv::doPosteriorDraws() {
	updateCoefPrec();
	sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
	updateCoef();
	updateCoefShrink();
	updateImpactPrec();
	latent_innov = y - x * coef_mat; // E_t before a
	updateImpact();
	chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
	updateState();
	updateStateVar();
	updateInitState();
	updateRecords();
}

Rcpp::List HorseshoeSv::returnRecords(int num_burn, int thin) const {
	Rcpp::List res = Rcpp::List::create(
		Rcpp::Named("alpha_record") = coef_record,
    Rcpp::Named("h_record") = lvol_record,
    Rcpp::Named("a_record") = contem_coef_record,
    Rcpp::Named("h0_record") = lvol_init_record,
    Rcpp::Named("sigh_record") = lvol_sig_record,
    Rcpp::Named("lambda_record") = local_record,
    Rcpp::Named("tau_record") = global_record,
    Rcpp::Named("kappa_record") = shrink_record
  );
	for (auto& record : res) {
		record = thin_record(record, num_iter, num_burn, thin);
	}
	return res;
}
