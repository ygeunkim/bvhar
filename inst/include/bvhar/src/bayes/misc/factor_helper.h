#ifndef BVHAR_BAYES_MISC_FACTOR_HELPER_H
#define BVHAR_BAYES_MISC_FACTOR_HELPER_H

#include "./helper.h"

namespace bvhar {

inline void draw_normal_factor(Eigen::Ref<Eigen::VectorXd> factor_t, Eigen::Ref<const Eigen::VectorXd> y,
											  			 Eigen::Ref<const Eigen::MatrixXd> factor_loading, Eigen::Ref<const Eigen::MatrixXd> sig_lower,
															 BVHAR_BHRNG& rng) {
	int size_factor = factor_t.size();
	Eigen::VectorXd res(size_factor);
	for (int i = 0; i < size_factor; i++) {
		res[i] = normal_rand(rng);
  }
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(
		(factor_loading.transpose() * sig_lower * sig_lower.transpose() * factor_loading + Eigen::MatrixXd::Identity(size_factor, size_factor)).selfadjointView<Eigen::Lower>()
	);
	Eigen::VectorXd post_mean = llt_of_prec.solve(
		sig_lower.transpose().triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(factor_loading.transpose()) *
		sig_lower.triangularView<Eigen::Lower>().solve(y)
	);
	factor_t = post_mean + llt_of_prec.matrixU().solve(res);
}

} // namespace bvhar

#endif // BVHAR_BAYES_MISC_FACTOR_HELPER_H