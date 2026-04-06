#ifndef BVHAR_BAYES_MISC_HELPER_H_H
#define BVHAR_BAYES_MISC_HELPER_H_H

#include "../../math/random.h"
#include "../../math/structural.h"
#include <limits>

namespace baecon {
namespace bvhar {

// Build std::set own-id and cross-id based on VectorXd
inline void set_grp_id(std::set<int>& own_id, std::set<int> cross_id, const Eigen::VectorXi& grp_own, const Eigen::VectorXi& grp_cross) {
	for (int i = 0; i < grp_own.size(); ++i) {
		own_id.insert(grp_own[i]);
	}
	for (int i = 0; i < grp_cross.size(); ++i) {
		cross_id.insert(grp_cross[i]);
	}
}

// inline void set_grp_id(std::set<int>& own_id, std::set<int> cross_id, bool& minnesota,
// 											 const Eigen::VectorXi& grp_own, const Eigen::VectorXi& grp_cross, const Eigen::MatrixXi& grp_mat) {
// 	minnesota = true;
// 	std::set<int> unique_grp(grp_mat.data(), grp_mat.data() + grp_mat.size());
// 	if (unique_grp.size() == 1) {
// 		minnesota = false;
// 	}
// 	for (int i = 0; i < grp_own.size(); ++i) {
// 		own_id.insert(grp_own[i]);
// 	}
// 	for (int i = 0; i < grp_cross.size(); ++i) {
// 		cross_id.insert(grp_cross[i]);
// 	}
// }

inline void cut_param(Eigen::Ref<Eigen::VectorXd> param) {
	param = param.array().isNaN().select(
		std::numeric_limits<double>::min(),
		param.array().isInf().select(
			std::numeric_limits<double>::max(),
			param
		)
	);
	param = param.cwiseMax(std::numeric_limits<double>::min()).cwiseMin(std::numeric_limits<double>::max());
}

// Building Lower Triangular Matrix
// 
// In MCMC, this function builds \eqn{L} given \eqn{a} vector.
// 
// @param dim Dimension (dim x dim) of L
// @param lower_vec Vector a
inline Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Identity(dim, dim);
  int id = 0;
  for (int i = 1; i < dim; i++) {
    res.row(i).segment(0, i) = lower_vec.segment(id, i);
    id += i;
  }
  return res;
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_HELPER_H_H