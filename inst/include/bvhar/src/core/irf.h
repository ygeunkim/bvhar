#ifndef BVHAR_CORE_IRF_H
#define BVHAR_CORE_IRF_H

#include "../math/structural.h"
#include "./omp.h"

namespace baecon {
namespace bvhar {

class ImpulseResponse;
class IrfRun;

/**
 * @brief Base class for impulse response function
 * 
 */
class ImpulseResponse {
public:
	ImpulseResponse(int lag_max, int lag, bool orthogonal = true)
	: step(lag_max), lag(lag), orthogonal(orthogonal) {}
	virtual ~ImpulseResponse() = default;

	/**
	 * @brief Compute IRF
	 * 
	 */
	virtual void computeIrf() {}

protected:
	int step, lag;
	bool orthogonal;
};

/**
 * @brief Base class for IRF runner
 * 
 */
class IrfRun {
public:
	IrfRun() : debug_logger(BVHAR_DEBUG_LOGGER("IrfRun")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Constructor");
	}
	
	virtual ~IrfRun() = default;

	virtual void computeIrf() {}
	
	// std::vector<Eigen::MatrixXd> returnIrf() {
	// 	computeIrf();
	// 	return density_irf;
	// }

protected:
	std::shared_ptr<spdlog::logger> debug_logger;

// private:
// 	std::vector<std::unique_ptr<ImpulseResponse>> irf_ptr;
// 	std::vector<Eigen::MatrixXd> density_irf;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_CORE_IRF_H