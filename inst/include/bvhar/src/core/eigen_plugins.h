#ifndef BVHAR_CORE_EIGEN_PLUGINS_H
#define BVHAR_CORE_EIGEN_PLUGINS_H

inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> unique() const {
  std::vector<Scalar> v(derived().data(), derived().data() + derived().size());
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(v.data(), v.size()).eval();
}

#endif // BVHAR_CORE_EIGEN_PLUGINS_H