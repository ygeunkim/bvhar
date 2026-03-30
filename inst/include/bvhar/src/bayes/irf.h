#ifndef BVHAR_BAYES_IRF_H
#define BVHAR_BAYES_IRF_H

#include "../core/irf.h"

namespace baecon {
namespace bvhar {

class McmcIrf;
class McmcIrfRun;

/**
 * @brief IRF class for MCMC
 * 
 */
class McmcIrf : public ImpulseResponse {
public:
	McmcIrf(int lag_max, int lag, int num_sim, bool orthogonal = true)
	: ImpulseResponse(lag_max, lag, orthogonal),
		num_sim(num_sim) {}

	virtual ~McmcIrf() = default;
	
	void computeIrf() override {
		for (int i = 0; i < num_sim; ++i) {
			updateParams(i);
			updateMovingAverage(i);
		}
	}

	Eigen::MatrixXd returnIrfDensity() {
		computeIrf();
		return vma_record;
	}

protected:
	int num_sim;
	Eigen::MatrixXd vma_record;

	virtual void updateParams(const int i) = 0;

	virtual void updateMovingAverage(const int i) = 0;
};

/**
 * @brief IRF running class
 * 
 */
class McmcIrfRun : public IrfRun {
public:
	McmcIrfRun(int num_chains, int nthreads)
	: num_chains(num_chains), nthreads(nthreads),
		irf_ptr(num_chains), density_irf(num_chains) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcIrfRun Constructor: num_chains={}, nthreads={}",
			num_chains, nthreads
		);
	}
	
	virtual ~McmcIrfRun() = default;

	void computeIrf() override {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			density_irf[chain] = irf_ptr[chain]->returnIrfDensity();
		}
	}
	
	std::vector<Eigen::MatrixXd> returnIrf() {
		computeIrf();
		return density_irf;
	}

protected:
	int num_chains, nthreads;
	std::vector<std::unique_ptr<McmcIrf>> irf_ptr;
	std::vector<Eigen::MatrixXd> density_irf;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_IRF_H