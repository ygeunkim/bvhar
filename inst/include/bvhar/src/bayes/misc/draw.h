#ifndef BVHAR_BAYES_MISC_DRAW_H
#define BVHAR_BAYES_MISC_DRAW_H

#include "./coef_helper.h"
#include "./sv_helper.h"
#include "./factor_helper.h"
#include "./minn_helper.h"
#include "./ssvs_helper.h"
#include "./hs_helper.h"
#include "./ng_helper.h"
#include "./dl_helper.h"
#include "./gdp_helper.h"

namespace baecon {
namespace bvhar {

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options> thin_record(const Eigen::MatrixBase<Derived>& record, int num_iter, int num_burn, int thin) {
  if (thin == 1) {
    return record.bottomRows(num_iter - num_burn);
  }
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options> col_record(record.bottomRows(num_iter - num_burn));
  int num_res = (num_iter - num_burn + thin - 1) / thin; // nrow after thinning
  Eigen::Map<const Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options>, 0, Eigen::InnerStride<>> res(
    col_record.data(),
    num_res, record.cols(),
    Eigen::InnerStride<>(thin * col_record.innerStride())
  );
  return res;
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_DRAW_H