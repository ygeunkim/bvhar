#ifndef BVHAR_MATH_RNG_H
#define BVHAR_MATH_RNG_H

#include "../core/common.h"

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

// template <typename PtrVec>
// inline void update_rngbatch(PtrVec& models, const std::vector<BVHAR_BHRNG>& rng_vec) {
//   int num_rng = models.size();
//   for (size_t i = 0; i < num_rng; ++i) {
//     models[i]->updateRng(rng_vec[i]);
//   }
// }

// template <typename PtrVec>
// inline std::vector<BVHAR_BHRNG> get_rngbatch(const PtrVec& models) {
//   std::vector<BVHAR_BHRNG> out;
//   out.reserve(models.size());
//   for (const auto& p : models) {
//     out.push_back(p->getRng());
//   }
//   return out;
// }

// template <typename PtrVec>
// inline void update_rngbatch_2d(PtrVec& models, const std::vector<std::vector<BVHAR_BHRNG>>& rng_vec) {
//   int num_outer = models.size();
//   for (size_t i = 0; i < num_outer; ++i) {
// 		int num_inner = models[i].size();
// 		for (int j = 0; j < num_inner; ++j) {
// 			models[i][j]->updateRng(rng_vec[i][j]);
// 		}
//   }
// }

// template <typename PtrVec>
// inline std::vector<std::vector<BVHAR_BHRNG>> get_rngbatch_2d(const PtrVec& models) {
// 	int num_outer = models.size();
//   std::vector<std::vector<BVHAR_BHRNG>> out(num_outer);
// 	for (size_t i = 0; i < num_outer; ++i) {
// 		out[i].reserve(models[i].size());
// 		for (const auto& p : models[i]) {
// 			out[i].push_back(p->getRng());
// 		}
//   }
//   return out;
// }

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_MATH_RNG_H