#ifndef BAYESAUTOREG_H
#define BAYESAUTOREG_H

#include "bvharmcmc.h"
#include "bvharinterrupt.h"

namespace bvhar {

class McmcInterface;
template <typename PARAMS, typename INITS, typename MCMC> class McmcRun;

class McmcInterface {
public:
	virtual ~McmcInterface() = default;
	virtual LIST_OF_LIST returnMcmc() = 0;
};

template <
	typename PARAMS = RegParams, // RegParams or SvParams(2)
	typename INITS = typename std::conditional<
		std::is_same<PARAMS, RegParams>::value,
		LdltInits2,
		SvInits2
	>::type,
	typename MCMC = typename std::conditional<
		std::is_same<PARAMS, RegParams>::value,
		McmcReg2,
		McmcSv2
	>::type
>
class McmcRun : public McmcInterface {
public:
	McmcRun(
		int num_chains, int num_iter, int num_burn, int thin, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
    const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
    bool include_mean, const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads
	)
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), mcmc_objs(num_chains), res(num_chains) {
		switch (prior_type) {
			case 1: {
				MinnParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					LIST init_spec = param_init[i];
					// std::unique_ptr<RegInits> _inits;
					// if (CONTAINS(param_reg, "initial_mean")) {
					// 	_inits.reset(new SvInits2(_inits[i]));
					// } else {
					// 	_inits.reset(new LdltInits2(_inits[i]));
					// }
					INITS _inits(init_spec);
					mcmc_objs[i].reset(new MinnReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 2: {
				SsvsParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					LIST init_spec = param_init[i];
					SsvsInits2<INITS> _inits(init_spec);
					mcmc_objs[i].reset(new SsvsReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 3: {
				HorseshoeParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					LIST init_spec = param_init[i];
					HsInits2<INITS> _inits(init_spec);
					mcmc_objs[i].reset(new HorseshoeReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 4: {
				HierMinnParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior, param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					LIST init_spec = param_init[i];
					HierminnInits2<INITS> _inits(init_spec);
					mcmc_objs[i].reset(new HierMinnReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 5: {
				NgParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					LIST init_spec = param_init[i];
					NgInits2<INITS> _inits(init_spec);
					mcmc_objs[i].reset(new NgReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 6: {
				DlParams2<PARAMS> _params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					LIST init_spec = param_init[i];
					GlInits2<INITS> _inits(init_spec);
					mcmc_objs[i].reset(new DlReg2<MCMC>(_params, _inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
		}
	}
	virtual ~McmcRun() = default;
	LIST_OF_LIST returnMcmc() override {
		fit();
		return WRAP(res);
	}

protected:
	void runGibbs(int chain) {
		bvharprogress bar(num_iter, display_progress);
		bvharinterrupt();
		for (int i = 0; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mcmc_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			mcmc_objs[chain]->doPosteriorDraws();
			bar.update();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mcmc_objs[chain]->returnRecords(num_burn, thin);
		}
	}
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

private:
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<bvhar::McmcCta>> mcmc_objs;
	std::vector<LIST> res;
};

// inline std::unique_ptr<McmcInterface> initMcmcRun(
// 	int num_chains, int num_iter, int num_burn, int thin, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
// 	LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
//   const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
//   bool include_mean, const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads
// ) {
// 	std::unique_ptr<bvhar::McmcInterface> mcmc;
// 	if (param_reg.containsElementNamed("initial_mean")) {
// 		mcmc.reset(new bvhar::McmcRun<bvhar::SvParams2>(
// 			num_chains, num_iter, num_burn, thin, x, y,
// 			param_reg, param_prior, param_intercept, param_init, prior_type,
// 			grp_id, own_id, cross_id, grp_mat,
// 			include_mean, seed_chain, display_progress, nthreads
// 		));
// 	} else {
// 		mcmc.reset(new bvhar::McmcRun<bvhar::RegParams>(
// 			num_chains, num_iter, num_burn, thin, x, y,
// 			param_reg, param_prior, param_intercept, param_init, prior_type,
// 			grp_id, own_id, cross_id, grp_mat,
// 			include_mean, seed_chain, display_progress, nthreads
// 		));
// 	}
// 	return mcmc;
// }

}; // namespace bvhar

#endif // BAYESAUTOREG_H