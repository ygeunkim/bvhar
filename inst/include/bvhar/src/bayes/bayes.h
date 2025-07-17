/**
 * @file bayes.h
 * @author your name (you@domain.com)
 * @brief MCMC base class
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef BVHAR_BAYES_BAYES_H
#define BVHAR_BAYES_BAYES_H

#include "../core/common.h"
#include "../core/progress.h"
#include "../core/interrupt.h"

namespace bvhar {

struct McmcParams;
class McmcAlgo;
class McmcRun;

/**
 * @brief Base input for `McmcAlgo`
 * 
 */
struct McmcParams {
	int _iter;

	McmcParams(int num_iter)
	: _iter(num_iter) {}
};

/**
 * @brief Base class for MCMC algorithm
 * 
 * This class is a base class for various MCMC algorithms.
 * 
 */
class McmcAlgo {
public:
	McmcAlgo(const McmcParams& params, unsigned int seed)
	: num_iter(params._iter), mcmc_step(0), rng(seed), debug_logger(BVHAR_DEBUG_LOGGER("McmcAlgo")) {
    BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger,"Constructor: num_iter={}", num_iter);
	}
	virtual ~McmcAlgo() {
		BVHAR_DEBUG_DROP("McmcAlgo");
	}
	
	/**
	 * @brief MCMC warmup step
	 * 
	 */
	virtual void doWarmUp() = 0;

	/**
	 * @brief MCMC posterior sampling step
	 * 
	 */
	virtual void doPosteriorDraws() = 0;

	/**
	 * @brief Return posterior sampling records
	 * 
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return BVHAR_LIST `BVHAR_LIST` containing every MCMC draws
	 */
	virtual BVHAR_LIST returnRecords(int num_burn, int thin) = 0;

protected:
	std::mutex mtx;
	int num_iter;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain
	std::shared_ptr<spdlog::logger> debug_logger;

	/**
	 * @brief Increment the MCMC step
	 * 
	 */
	void addStep() { ++mcmc_step; }
};

/**
 * @brief Class that conducts MCMC
 * 
 */
class McmcRun {
public:
	McmcRun(int num_chains, int num_iter, int num_burn, int thin, bool display_progress, int nthreads)
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), mcmc_ptr(num_chains), res(num_chains) {}
	virtual ~McmcRun() = default;

	/**
	 * @brief Conduct multi-chain MCMC
	 * 
	 */
	void fit() {
		if (num_chains == 1) {
			runGibbs(0);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				runGibbs(chain);
			}
		}
	}

	/**
	 * @brief Conduct multi-chain MCMC and return MCMC records of every chain
	 * 
	 * @return BVHAR_LIST_OF_LIST `BVHAR_LIST_OF_LIST`
	 */
	BVHAR_LIST_OF_LIST returnRecords() {
		fit();
		return BVHAR_WRAP(res);
	}

protected:
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<McmcAlgo>> mcmc_ptr;
	std::vector<BVHAR_LIST> res;

	/**
	 * @brief Single chain MCMC
	 * 
	 * @param chain Chain id
	 */
	void runGibbs(int chain) {
		std::string log_name = fmt::format("Chain {}", chain + 1);
		auto logger = spdlog::get(log_name);
		if (logger == nullptr) {
			logger = SPDLOG_SINK_MT(log_name);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		int logging_freq = num_iter / 20; // 5 percent
		if (logging_freq == 0) {
			logging_freq = 1;
		}
		BVHAR_INIT_DEBUG(logger);
		bvharinterrupt();
		for (int i = 0; i < num_burn; ++i) {
			mcmc_ptr[chain]->doWarmUp();
			BVHAR_DEBUG_LOG(logger, "{} / {} (Warmup)", i + 1, num_iter);
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Warmup)", i + 1, num_iter);
			}
		}
		logger->flush();
		for (int i = num_burn; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
				logger->warn("User interrupt in {} / {}", i + 1, num_iter);
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mcmc_ptr[chain]->returnRecords(0, 1);
				}
				break;
			}
			mcmc_ptr[chain]->doPosteriorDraws();
			BVHAR_DEBUG_LOG(logger, "{} / {} (Sampling)", i + 1, num_iter);
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Sampling)", i + 1, num_iter);
			}
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mcmc_ptr[chain]->returnRecords(0, thin);
		}
		logger->flush();
		spdlog::drop(log_name);
	}
};

} // namespace bvhar

#endif // BVHAR_BAYES_BAYES_H