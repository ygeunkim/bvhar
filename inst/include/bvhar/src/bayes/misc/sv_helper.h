#ifndef BVHAR_BAYES_MISC_SV_HELPER_H_H
#define BVHAR_BAYES_MISC_SV_HELPER_H_H

#include "./helper.h"

namespace bvhar {

// Generating log-volatilities in MCMC
// 
// In MCMC, this function samples log-volatilities \eqn{h_{it}} vector using auxiliary mixture sampling
// 
// @param sv_vec log-volatilities vector
// @param init_sv Initial log-volatility
// @param sv_sig Variance of log-volatilities
// @param latent_vec Auxiliary residual vector
inline void varsv_ht(Eigen::Ref<Eigen::VectorXd> sv_vec, double init_sv,
										 double sv_sig, Eigen::Ref<Eigen::VectorXd> latent_vec, BVHAR_BHRNG& rng) {
  int num_design = sv_vec.size(); // h_i1, ..., h_in for i = 1, .., k
  // 7-component normal mixutre
	using Vector7d = Eigen::Matrix<double, 7, 1>;
	using Matrix7d = Eigen::Matrix<double, Eigen::Dynamic, 7>;
	Vector7d pj(0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575); // p_t
	Vector7d muj(-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819); // mu_t
  muj.array() -= 1.2704;
	Vector7d sigj(5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261); // sig_t^2
	Vector7d sdj = sigj.cwiseSqrt();
  Eigen::VectorXi binom_latent(num_design);
  Eigen::VectorXd ds(num_design); // (mu_st - 1.2704)
  Eigen::MatrixXd inv_sig_s = Eigen::MatrixXd::Zero(num_design, num_design); // diag(1 / sig_st^2)
  Eigen::VectorXd inv_method(num_design); // inverse transform method
	Matrix7d mixture_pdf(num_design, 7);
	Matrix7d mixture_cumsum = Matrix7d::Zero(num_design, 7);
  for (int i = 0; i < num_design; i++) {
		inv_method[i] = unif_rand(0, 1, rng);
  }
  for (int i = 0; i < 7; i++) {
    mixture_pdf.col(i) = (-((latent_vec.array() - sv_vec.array() - muj[i]).array() / sdj[i]).array().square() / 2).exp() * pj[i] / (sdj[i] * sqrt(2 * M_PI));
  }
  mixture_pdf.array().colwise() /= mixture_pdf.rowwise().sum().array();
  for (int i = 0; i < 7; i++) {
    mixture_cumsum.block(0, i, num_design, 7 - i) += mixture_pdf.col(i).rowwise().replicate(7 - i);
  }
  binom_latent.array() = 7 - (inv_method.rowwise().replicate(7).array() < mixture_cumsum.array()).cast<int>().rowwise().sum().array(); // 0 to 6 for indexing
  Eigen::MatrixXd diff_mat = Eigen::MatrixXd::Identity(num_design, num_design);
  for (int i = 0; i < num_design - 1; i++) {
    ds[i] = muj[binom_latent[i]];
    inv_sig_s(i, i) = 1 / sigj[binom_latent[i]];
    diff_mat(i + 1, i) = -1;
  }
  ds[num_design - 1] = muj[binom_latent[num_design - 1]];
  inv_sig_s(num_design - 1, num_design - 1) = 1 / sigj[binom_latent[num_design - 1]];
  Eigen::MatrixXd HtH = diff_mat.transpose() * diff_mat;
  Eigen::VectorXd res(num_design);
  for (int i = 0; i < num_design; i++) {
		res[i] = normal_rand(rng);
  }
  Eigen::MatrixXd post_sig = HtH / sv_sig + inv_sig_s;
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(post_sig);
  Eigen::VectorXd post_mean = lltOfscale.solve(
    HtH * init_sv * Eigen::VectorXd::Ones(num_design) / sv_sig + inv_sig_s * (latent_vec - ds)
  );
	sv_vec = post_mean + lltOfscale.matrixU().solve(res);
}

// Generating sig_h in MCMC
// 
// In MCMC, this function samples \eqn{\sigma_h^2} in VAR-SV.
// 
// @param shp Prior shape of sigma
// @param scl Prior scale of sigma
// @param init_sv Initial log volatility
// @param h1 Time-varying h1 matrix
inline void varsv_sigh(Eigen::VectorXd& sv_sig, Eigen::VectorXd& shp, Eigen::VectorXd& scl,
											 Eigen::VectorXd& init_sv, Eigen::MatrixXd& h1, BVHAR_BHRNG& rng) {
  int dim = init_sv.size();
  int num_design = h1.rows();
  Eigen::MatrixXd h_slide(num_design, dim); // h_ij, j = 0, ..., n - 1
  h_slide.row(0) = init_sv;
  h_slide.bottomRows(num_design - 1) = h1.topRows(num_design - 1);
  for (int i = 0; i < dim; i++) {
    sv_sig[i] = 1 / gamma_rand(
      shp[i] + num_design / 2,
			1 / (scl[i] + (h1.array() - h_slide.array()).square().sum() / 2),
			rng
    );
  }
}

// Generating h0 in MCMC
// 
// In MCMC, this function samples h0 in VAR-SV.
// 
// @param prior_mean Prior mean vector of h0.
// @param prior_prec Prior precision of h0.
// @param init_sv Initial log volatility
// @param h1 h1
// @param sv_prec Precision of log volatility
inline void varsv_h0(Eigen::Ref<Eigen::VectorXd> h0, Eigen::Ref<Eigen::VectorXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prior_prec,
										 Eigen::Ref<const Eigen::VectorXd> h1, Eigen::Ref<const Eigen::VectorXd> sv_prec, BVHAR_BHRNG& rng) {
  int dim = h1.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; ++i) {
		res[i] = normal_rand(rng);
  }
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(prior_prec.asDiagonal() + sv_prec.asDiagonal());
  Eigen::VectorXd post_mean = llt_of_prec.solve(prior_prec.cwiseProduct(prior_mean) + sv_prec.cwiseProduct(h1));
	h0 = post_mean + llt_of_prec.matrixU().solve(res);
}

} // namespace bvhar

#endif // BVHAR_BAYES_MISC_SV_HELPER_H_H