#include "mcmcsv.h"

McmcSv::McmcSv(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::VectorXd prior_sig_shp, Eigen::VectorXd prior_sig_scl, Eigen::VectorXd prior_init_mean, Eigen::MatrixXd prior_init_prec)
: x(x), y(y),
	prior_sig_shp(prior_sig_shp), prior_sig_scl(prior_sig_scl),
	prior_init_mean(prior_init_mean), prior_init_prec(prior_init_prec),
	dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
	chol_lower(Eigen::MatrixXd::Zero(dim, dim)),
	ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
	prior_mean_j(Eigen::VectorXd::Zero(dim_design)),
	prior_prec_j(Eigen::MatrixXd::Identity(dim_design, dim_design)),
	response_contem(Eigen::VectorXd::Zero(num_design)),
	sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
	contem_id(0) {
  num_lowerchol = dim * (dim - 1) / 2;
  num_coef = dim * dim_design;
	coef_mat = (x.transpose() * x).llt().solve(x.transpose() * y);
	contem_coef = Eigen::VectorXd(num_lowerchol);
	latent_innov = y - x * coef_mat;
	lvol_init = latent_innov.transpose().array().square().rowwise().mean().log();
	lvol_draw = lvol_init.replicate(num_design, 1);
	lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
  coef_j = coef_mat;
  coef_j.col(0) = Eigen::VectorXd::Zero(dim_design);
}

void McmcSv::UpdateCoef(Eigen::MatrixXd prior_alpha_mean, Eigen::MatrixXd prior_alpha_prec) {
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
}

void McmcSv::UpdateState() {
	latent_innov = y - x * coef_mat;
  ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
	ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
	for (int t = 0; t < dim; t++) {
		lvol_draw.col(t) = varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t));
	}
}

void McmcSv::UpdateImpact(Eigen::MatrixXd prior_chol_mean, Eigen::MatrixXd prior_chol_prec) {
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
