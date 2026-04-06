#ifndef BVHAR_MATH_RNG_H
#define BVHAR_MATH_RNG_H

#include "../core/common.h"
#include <utility>

namespace baecon {
namespace bvhar {

class RngState;

/**
 * @brief RNG state
 * 
 */
class RngState {
public:
	RngState() {}

	RngState(unsigned int seed) : rng(seed) {}

	RngState(BVHAR_BHRNG& rng) : rng(rng) {}

	virtual ~RngState() = default;
	
	/**
	 * @brief Update rng object with external one
	 * 
	 * @param external_rng rng object
	 */
	void updateRng(const BVHAR_BHRNG& external_rng) {
		rng = external_rng;
	}

	/**
	 * @brief Get the Rng state
	 * 
	 * Use the rng of this object so that the random works outside can update the rng member.
	 * 
	 * @return BVHAR_BHRNG& 
	 */
	BVHAR_BHRNG& getRng() {
		return rng;
	}

	/**
	 * @brief Get read-only Rng state
	 * @overload getRng()
	 * @return const BVHAR_BHRNG& 
	 */
	const BVHAR_BHRNG& getRng() const {
		return rng;
	}

protected:
	BVHAR_BHRNG rng; // RNG instance
};

/**
 * @brief Update the rng batch recursively for nested containers
 * 
 * @tparam Ptr Type of the model container including smart pointer
 * @param model_ptr 
 * @param rng 
 */
template <typename Ptr>
inline void update_rng_batch(Ptr& model_ptr, const BVHAR_BHRNG& rng) {
  model_ptr->updateRng(rng);
}

/**
 * @overload template <typename Ptr> void update_rng_batch(Ptr&, const BVHAR_BHRNG&)
 * @tparam RngVec Type of rng vector.
 */
template <typename Ptr, typename RngVec>
inline void update_rng_batch(std::vector<Ptr>& models, const std::vector<RngVec>& rngs) {
	int outer_size = models.size();
	// Add size check here
  for (int i = 0; i < outer_size; ++i) {
    update_rng_batch(models[i], rngs[i]);
  }
}

/**
 * @brief Get the rng batch recursively from nested containers
 * 
 * @tparam Ptr Type of the model container including smart pointer
 * @param model_ptr 
 * @return BVHAR_BHRNG 
 */
template <typename Ptr>
inline BVHAR_BHRNG get_rng_batch(const Ptr& model_ptr) {
  return model_ptr->getRng();
}

/**
 * @overload template <typename Ptr> BVHAR_BHRNG get_rng_batch(const Ptr&)
 */
template <typename Ptr>
inline auto get_rng_batch(const std::vector<Ptr>& models)
#if defined(BVHAR_USE_RCPP) || defined(BVHAR_USE_PYBIND11)
	-> std::vector<decltype(get_rng_batch(std::declval<Ptr>()))>
#endif
	{ // Return type can be skipped in C++14 and C++17
  std::vector<decltype(get_rng_batch(std::declval<Ptr>()))> rng_vec;
  rng_vec.reserve(models.size());
  for (const auto& p : models) {
    rng_vec.push_back(get_rng_batch(p));
  }
  return rng_vec;
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_MATH_RNG_H