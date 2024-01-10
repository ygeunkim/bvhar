#include "mcmcssvs.h"

McmcSsvs::McmcSsvs(
	int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	const Eigen::VectorXd& init_coef, const Eigen::VectorXd& init_chol_diag, const Eigen::VectorXd& init_chol_upper,
  const Eigen::VectorXd& init_coef_dummy, const Eigen::VectorXd& init_chol_dummy,
  const Eigen::VectorXd& coef_spike, const Eigen::VectorXd& coef_slab, const Eigen::VectorXd& coef_slab_weight,
  const Eigen::VectorXd& shape, const Eigen::VectorXd& rate,
  const double& coef_s1, const double& coef_s2,
  const Eigen::VectorXd& chol_spike, const Eigen::VectorXd& chol_slab, const Eigen::VectorXd& chol_slab_weight,
  const double& chol_s1, const double& chol_s2,
  const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
  const Eigen::VectorXd& mean_non, const double& sd_non, bool include_mean, bool init_gibbs
)
: num_iter(num_iter), x(x), y(y),
	dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
	num_coef(dim * dim_design), num_upperchol(dim * (dim - 1) / 2),
	coef_spike(coef_spike), coef_slab(coef_slab), chol_spike(chol_spike), chol_slab(chol_slab),
	shape(shape), rate(rate),
	coef_s1(coef_s1), coef_s2(coef_s2), chol_s1(chol_s1), chol_s2(chol_s2),
	prior_mean_non(mean_non), prior_sd_non(sd_non), prior_sd(Eigen::VectorXd::Zero(num_coef)),
	grp_id(grp_id), grp_mat(grp_mat), grp_vec(vectorize_eigen(grp_mat)), num_grp(grp_id.size()),
	include_mean(include_mean), mcmc_step(0) {
	num_restrict = num_coef - dim; // number of restricted coefs: dim^2 p vs dim^2 p - dim (if no constant)
	if (!include_mean) {
    num_restrict += dim; // always dim^2 p
  }
	coef_mean = Eigen::VectorXd::Zero(num_restrict); // zero prior mean for restricted coefficient
	prior_mean = Eigen::VectorXd::Zero(num_coef);
	if (include_mean) {
    for (int j = 0; j < dim; j++) {
      prior_mean.segment(j * dim_design, num_restrict / dim) = coef_mean.segment(j * num_restrict / dim, num_restrict / dim);
      prior_mean[j * dim_design + num_restrict / dim] = prior_mean_non[j];
    }
  } else {
    prior_mean = coef_mean;
  }
	coef_mixture_mat = Eigen::VectorXd(num_restrict);
	chol_mixture_mat = Eigen::VectorXd(num_restrict);
	slab_weight = Eigen::VectorXd(num_restrict);
	slab_weight_mat = Eigen::MatrixXd(num_restrict / dim, dim);
	gram = x.transpose() * x;
	coef_ols = gram.llt().solve(x.transpose() * y);
	coef_vec = vectorize_eigen(coef_ols);
	chol_ols = ((y - x * coef_ols).transpose() * (y - x * coef_ols) / (num_design - dim_design)).llt().matrixU();
	coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef);
	coef_dummy_record = Eigen::MatrixXd::Zero(num_iter + 1, num_restrict);
	coef_weight_record = Eigen::MatrixXd::Zero(num_iter + 1, num_grp);
	chol_diag_record = Eigen::MatrixXd::Zero(num_iter + 1, dim);
	chol_upper_record = Eigen::MatrixXd::Zero(num_iter + 1, num_upperchol);
	chol_dummy_record = Eigen::MatrixXd::Zero(num_iter + 1, num_upperchol);
	chol_weight_record = Eigen::MatrixXd::Zero(num_iter + 1, num_upperchol);
	chol_factor_record = Eigen::MatrixXd::Zero(dim * (num_iter + 1), dim);
	coef_weight = coef_slab_weight;
	chol_weight = chol_slab_weight;
	if (init_gibbs) {
		coef_draw = init_coef;
		coef_dummy = init_coef_dummy;
		chol_diag = init_chol_diag;
		chol_coef = init_chol_upper;
		chol_dummy = init_chol_dummy;
		chol_factor = build_chol(init_chol_diag, init_chol_upper);
	} else {
		coef_draw = coef_vec;
		coef_dummy = Eigen::VectorXd::Ones(num_restrict);
		chol_diag = chol_ols.diagonal();
		chol_coef = Eigen::VectorXd::Zero(num_upperchol);
		for (int i = 1; i < dim; i++) {
			chol_coef.segment(i * (i - 1) / 2, i) = chol_ols.block(0, i, i, 1);
		}
		chol_dummy = Eigen::VectorXd::Ones(num_upperchol);
		chol_factor = chol_ols;
	}
	coef_record.row(0) = coef_draw;
	coef_dummy_record.row(0) = coef_dummy;
	chol_diag_record.row(0) = chol_diag;
	chol_upper_record.row(0) = chol_diag;
	chol_dummy_record.row(0) = chol_dummy;
	chol_factor_record.topLeftCorner(dim, dim) = chol_ols;
	coef_mat = unvectorize(coef_draw, dim_design, dim);
	sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
}

void McmcSsvs::addStep() {
	mcmc_step++;
}

void McmcSsvs::updateChol() {
	chol_mixture_mat = build_ssvs_sd(chol_spike, chol_slab, chol_dummy);
	ssvs_chol_diag(chol_diag, sse_mat, chol_mixture_mat, shape, rate, num_design);
	ssvs_chol_off(chol_coef, sse_mat, chol_diag, chol_mixture_mat);
	chol_factor = build_chol(chol_diag, chol_coef);
	chol_upper_record.row(mcmc_step) = chol_coef;
	chol_diag_record.row(mcmc_step) = chol_diag;
	chol_factor_record.block(mcmc_step * dim, 0, dim, dim) = chol_factor;
}

void McmcSsvs::updateCholDummy() {
	ssvs_dummy(chol_dummy, chol_coef, chol_slab, chol_spike, chol_weight);
	ssvs_weight(chol_weight, chol_dummy, chol_s1, chol_s2);
	chol_dummy_record.row(mcmc_step) = chol_dummy;
	chol_weight_record.row(mcmc_step) = chol_weight;
}

void McmcSsvs::updateCoef() {
	coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy);
	if (include_mean) {
		for (int j = 0; j < dim; j++) {
			prior_sd.segment(j * dim_design, num_restrict / dim) = coef_mixture_mat.segment(j * num_restrict / dim, num_restrict / dim);
      prior_sd[j * dim_design + num_restrict / dim] = prior_sd_non;
		}
	} else {
		prior_sd = coef_mixture_mat;
	}
	ssvs_coef(coef_draw, prior_mean, prior_sd, gram, coef_vec, chol_factor);
	coef_mat = unvectorize(coef_draw, dim_design, dim);
	sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
	coef_record.row(mcmc_step) = coef_draw;
}

void McmcSsvs::updateCoefDummy() {
	for (int j = 0; j < num_grp; j++) {
		slab_weight_mat = (grp_mat.array() == grp_id[j]).select(
      coef_weight.segment(j, 1).replicate(num_restrict / dim, dim),
      slab_weight_mat
    );
	}
	slab_weight = vectorize_eigen(slab_weight_mat);
	ssvs_dummy(
		coef_dummy,
		vectorize_eigen(coef_mat.topRows(num_restrict / dim)),
		coef_slab,
		coef_spike,
		slab_weight
	);
	ssvs_mn_weight(coef_weight, grp_vec, grp_id, coef_dummy, coef_s1, coef_s2);
	coef_dummy_record.row(mcmc_step) = coef_dummy;
	coef_weight_record.row(mcmc_step) = coef_weight;
}

void McmcSsvs::doPosteriorDraws() {
	updateChol();
	updateCholDummy();
	updateCoef();
	updateCoefDummy();
}

Rcpp::List McmcSsvs::returnRecords(int num_burn) const {
	return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("eta_record") = chol_upper_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("psi_record") = chol_diag_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("omega_record") = chol_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("chol_record") = chol_factor_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("p_record") = coef_weight_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("q_record") = chol_weight_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("ols_coef") = coef_ols,
    Rcpp::Named("ols_cholesky") = chol_ols
  );
}
