#ifndef BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H
#define BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H

#include "./config.h"

namespace baecon {
namespace bvhar {

class ShrinkageUpdater;
// Shrinkage priors
class MinnUpdater;
class HierminnUpdater;
class SsvsUpdater;
template <bool isGroup> class HorseshoeUpdater;
template <bool isGroup> class NgUpdater;
template <bool isGroup> class DlUpdater;
template <bool isGroup> class GdpUpdater;

/**
 * @brief Draw class for shrinkage priors
 * 
 */
class ShrinkageUpdater {
public:
	ShrinkageUpdater(int num_iter, const ShrinkageParams& params, const ShrinkageInits& inits) {}
	virtual ~ShrinkageUpdater() = default;

	/**
	 * @brief Build coefficient prior mean structure
	 * 
	 * @param prior_alpha_mean Coefficient prior mean
	 */
	virtual void initCoefMean(Eigen::Ref<Eigen::VectorXd> prior_alpha_mean) {}

	/**
	 * @brief Build coefficient prior precision structure
	 * 
	 * @param prior_alpha_prec Coefficient prior precision
	 * @param grp_vec Group vector
	 * @param cross_id Cross id
	 */
	virtual void initCoefPrec(Eigen::Ref<Eigen::VectorXd> prior_alpha_prec, Eigen::VectorXi& grp_vec, std::set<int>& cross_id) {}

	/**
	 * @brief Build contemporaneous coefficient precision structure
	 * 
	 * @param prior_chol_prec Contemporaneous coefficient prior precision
	 */
	virtual void initImpactPrec(Eigen::Ref<Eigen::VectorXd> prior_chol_prec) {}

	/**
	 * @brief Draw precision of coefficient based on each shrinkage priors
	 * 
	 * @param prior_alpha_prec Prior precision
	 * @param coef_vec Coefficient vector
	 * @param num_grp Group number
	 * @param grp_vec Group vector
	 * @param grp_id Group id
	 * @param rng RNG
	 */
	virtual void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) {}

	/**
	 * @brief Draw precision of contemporaneous coefficient based on each shrinkage priors
	 * 
	 * @param prior_chol_prec Prior precision
	 * @param contem_coef Contemporaneous coefficient
	 * @param rng RNG
	 */
	virtual void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) {}

	/**
	 * @brief Save MCMC records
	 * 
	 */
	virtual void updateRecords(int id) {}

	/**
	 * @brief Append coefficient shrinkage prior's parameter record to the result `BVHAR_LIST`
	 * 
	 * @param list Contains MCMC record result
	 */
	virtual void appendCoefRecords(BVHAR_LIST& list) {}

	/**
	 * @brief Append contemporaneous coefficient shrinkage prior's parameter record to the result `BVHAR_LIST`
	 * 
	 * @param list Contains MCMC record result
	 */
	virtual void appendContemRecords(BVHAR_LIST& list) {}
};

/**
 * @brief Minnesota prior
 * 
 */
class MinnUpdater : public ShrinkageUpdater {
public:
	MinnUpdater(int num_iter, const MinnParams& params, const ShrinkageInits& inits)
	: ShrinkageUpdater(num_iter, params, inits), prior_mean(params._prior_mean), prior_prec(params._prior_prec) {}
	virtual ~MinnUpdater() = default;
	
	void initCoefMean(Eigen::Ref<Eigen::VectorXd> prior_alpha_mean) override {
		prior_alpha_mean = prior_mean;
		prior_mean.resize(0);
	}

	void initCoefPrec(Eigen::Ref<Eigen::VectorXd> prior_alpha_prec, Eigen::VectorXi& grp_vec, std::set<int>& cross_id) override {
		prior_alpha_prec = prior_prec;
		prior_prec.resize(0);
	}

private:
	Eigen::VectorXd prior_mean, prior_prec;
};

/**
 * @brief Hierarchical Minnesota prior
 * 
 */
class HierminnUpdater : public ShrinkageUpdater {
public:
	HierminnUpdater(int num_iter, const HierminnParams& params, const HierminnInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		prior_mean(params._prior_mean), prior_prec(params._prior_prec),
		grid_size(params._grid_size),
		own_shape(params._shape), own_rate(params._rate),
		// cross_shape(params.shape), cross_rate(params.rate),
		own_lambda(inits._own_lambda), cross_lambda(inits._cross_lambda) {}
	virtual ~HierminnUpdater() = default;

	void initCoefMean(Eigen::Ref<Eigen::VectorXd> prior_alpha_mean) override {
		prior_alpha_mean = prior_mean;
	}

	void initCoefPrec(Eigen::Ref<Eigen::VectorXd> prior_alpha_prec, Eigen::VectorXi& grp_vec, std::set<int>& cross_id) override {
		prior_alpha_prec = prior_prec;
		prior_alpha_prec.array() /= own_lambda;
		for (int i = 0; i < prior_alpha_prec.size(); ++i) {
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec[i] /= cross_lambda; // nu
			}
		}
		prior_prec.resize(0);
	}

	void initImpactPrec(Eigen::Ref<Eigen::VectorXd> prior_chol_prec) override {
		prior_chol_prec.array() /= own_lambda; // divide because it is precision
		prior_prec.resize(0);
	}

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			coef_vec, prior_mean, prior_alpha_prec,
			rng
		);
		minnesota_nu_griddy(
			cross_lambda, grid_size,
			coef_vec, prior_mean, prior_alpha_prec,
			grp_vec, grp_id, rng
		);
	}
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			contem_coef, prior_mean, prior_chol_prec,
			rng
		);
	}

private:
	Eigen::VectorXd prior_mean, prior_prec;
	int grid_size;
	double own_shape, own_rate;
	double own_lambda, cross_lambda;
};

/**
 * @brief Stochastic Search Variable Selection (SSVS) prior
 * 
 */
class SsvsUpdater : public ShrinkageUpdater {
public:
	SsvsUpdater(int num_iter, const SsvsParams& params, const SsvsInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		grid_size(params._grid_size),
		ig_shape(params._slab_shape), ig_scl(params._slab_scl), s1(params._s1), s2(params._s2),
		spike_scl(inits._spike_scl), dummy(inits._dummy), weight(inits._weight), slab(inits._slab),
		slab_weight(Eigen::VectorXd::Ones(slab.size())),
		dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, dummy.size())),
		slab_record(Eigen::MatrixXd::Ones(num_iter + 1, slab.size())),
		weight_record(Eigen::MatrixXd::Zero(num_iter + 1, weight.size())) {}
	virtual ~SsvsUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		ssvs_local_slab(slab, dummy, coef_vec, ig_shape, ig_scl, spike_scl, rng);
		for (int j = 0; j < num_grp; ++j) {
			slab_weight = (grp_vec.array() == grp_id[j]).select(
				weight[j],
				slab_weight
			);
		}
		ssvs_scl_griddy(spike_scl, grid_size, coef_vec, dummy, slab, rng);
		ssvs_dummy(dummy, coef_vec, slab, spike_scl * slab, slab_weight, rng);
		ssvs_mn_weight(weight, grp_vec, grp_id, dummy, s1, s2, rng);
		prior_alpha_prec.array() = 1 / (spike_scl * (1 - dummy.array()) * slab.array() + dummy.array() * slab.array());
	}
	
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		ssvs_local_slab(slab, dummy, contem_coef, ig_shape, ig_scl, spike_scl, rng);
		ssvs_scl_griddy(spike_scl, grid_size, contem_coef, slab, dummy, rng);
		ssvs_dummy(dummy, contem_coef, slab, spike_scl * slab, weight, rng);
		ssvs_weight(weight, dummy, s1[0], s2[0], rng);
		// prior_chol_prec = 1 / build_ssvs_sd(spike_scl * slab, slab, dummy).array().square();
		prior_chol_prec = 1 / (spike_scl * (1 - dummy.array()) * slab.array() + dummy.array() * slab.array());
	}

	void updateRecords(int id) override {
		dummy_record.row(id) = dummy;
		slab_record.row(id) = slab;
		weight_record.row(id) = weight;
	}

	void appendCoefRecords(BVHAR_LIST& list) override {
		list["gamma_record"] = dummy_record;
		list["tau_record"] = slab_record;
	}

	void appendContemRecords(BVHAR_LIST& list) override {
		list["omega_record"] = dummy_record;
		list["kappa_record"] = slab_record;
	}

private:
	int grid_size;
	double ig_shape, ig_scl; // IG hyperparameter for spike sd
	Eigen::VectorXd s1, s2; // Beta hyperparameter
	double spike_scl; // scaling factor between 0 and 1: spike_sd = c * slab_sd
	Eigen::VectorXd dummy;
	Eigen::VectorXd weight;
	Eigen::VectorXd slab;
	Eigen::VectorXd slab_weight; // pij vector
	Eigen::MatrixXd dummy_record, slab_record, weight_record;
};

/**
 * @brief Horseshoe prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class HorseshoeUpdater : public ShrinkageUpdater {
public:
	HorseshoeUpdater(int num_iter, const ShrinkageParams& params, const HorseshoeInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		shrink_fac(Eigen::VectorXd::Zero(local_lev.size())),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		latent_group(Eigen::VectorXd::Zero(group_lev.size())),
		latent_global(0.0),
		coef_var(Eigen::VectorXd::Ones(local_lev.size())),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, local_lev.size())),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, group_lev.size())),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, shrink_fac.size())) {}
	virtual ~HorseshoeUpdater() = default;

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec, 1, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		horseshoe_latent(latent_local, local_lev, rng);
		// using is_group = std::integral_constant<bool, isGroup>;
		// if (is_group::value) {
		BVHAR_IF_CONSTEXPR(isGroup) {
			horseshoe_latent(latent_global, global_lev, rng);
			global_lev = horseshoe_global_sparsity(latent_global, coef_var.array() * local_lev.array(), coef_vec, 1, rng);
		}
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec, global_lev, rng);
		prior_alpha_prec = 1 / (global_lev * coef_var.array() * local_lev.array());
		shrink_fac = 1 / (1 + prior_alpha_prec.array());
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		coef_var = group_lev.replicate(1, prior_chol_prec.size()).reshaped();
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, contem_coef, 1, rng);
		group_lev[0] = horseshoe_global_sparsity(latent_group[0], latent_local, contem_coef, 1, rng);
		prior_chol_prec.setZero();
		prior_chol_prec = 1 / (coef_var.array() * local_lev.array());
	}

	void updateRecords(int id) override {
		shrink_record.row(id) = shrink_fac;
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	void appendCoefRecords(BVHAR_LIST& list) override {
		list["lambda_record"] = local_record;
		list["eta_record"] = group_record;
		list["tau_record"] = global_record;
		list["kappa_record"] = shrink_record;
	}

	void appendContemRecords(BVHAR_LIST& list) override {
		list["xi_record"] = local_record;
		list["zeta_record"] = group_record;
	}

private:
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd global_record;
	Eigen::MatrixXd local_record, group_record, shrink_record;
};

/**
 * @brief Normal-Gamma prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class NgUpdater : public ShrinkageUpdater {
public:
	NgUpdater(int num_iter, const NgParams& params, const NgInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		mh_sd(params._mh_sd),
		group_shape(params._group_shape), group_scl(params._group_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		local_shape(inits._local_shape),
		local_shape_fac(Eigen::VectorXd::Ones(inits._local.size())),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		coef_var(Eigen::VectorXd::Ones(local_lev.size())),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, local_lev.size())),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, group_lev.size())) {}
	virtual ~NgUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
		for (int j = 0; j < num_grp; ++j) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
			local_shape_fac = (grp_vec.array() == grp_id[j]).select(
				local_shape[j],
				local_shape_fac
			);
		}
		// using is_group = std::integral_constant<bool, isGroup>;
		// if (is_group::value) {
		BVHAR_IF_CONSTEXPR(isGroup) {
			global_lev = ng_global_sparsity(local_lev.array() / coef_var.array(), local_shape_fac, global_shape, global_scl, rng);
		}
		ng_local_sparsity(local_lev, local_shape_fac, coef_vec, global_lev * coef_var, rng);
		prior_alpha_prec = 1 / local_lev.array();
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		local_shape[0] = ng_shape_jump(local_shape[0], local_lev, group_lev[0], mh_sd, rng);
		group_lev[0] = ng_global_sparsity(local_lev, local_shape[0], group_shape, group_scl, rng);
		ng_local_sparsity(coef_var, local_shape[0], contem_coef, group_lev.replicate(1, prior_chol_prec.size()).reshaped(), rng);
		prior_chol_prec = 1 / local_lev.array();
	}

	void updateRecords(int id) override {
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	void appendCoefRecords(BVHAR_LIST& list) override {
		list["lambda_record"] = local_record;
		list["eta_record"] = group_record;
		list["tau_record"] = global_record;
	}

	void appendContemRecords(BVHAR_LIST& list) override {
		list["xi_record"] = local_record;
		list["zeta_record"] = group_record;
	}

private:
	double mh_sd;
	double group_shape, group_scl, global_shape, global_scl;
	Eigen::VectorXd local_shape, local_shape_fac;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd global_record;
	Eigen::MatrixXd local_record, group_record;
};

/**
 * @brief Dirichlet-Laplace prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class DlUpdater : public ShrinkageUpdater {
public:
	DlUpdater(int num_iter, const DlParams& params, const HorseshoeInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		dir_concen(.5), shape(params._shape), scl(params._scl), grid_size(params._grid_size),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		coef_var(Eigen::VectorXd::Zero(local_lev.size())),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, local_lev.size())),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, group_lev.size())) {}
	virtual ~DlUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		dl_mn_sparsity(group_lev, grp_vec, grp_id, global_lev, local_lev, shape, scl, coef_vec, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		dl_dir_griddy(dir_concen, grid_size, local_lev, global_lev, rng);
		dl_local_sparsity(local_lev, dir_concen, coef_vec.array() * coef_var.array(), rng);
		// using is_group = std::integral_constant<bool, isGroup>;
		// if (is_group::value) {
		BVHAR_IF_CONSTEXPR(isGroup) {
			global_lev = dl_global_sparsity(local_lev.array() / coef_var.array(), dir_concen, coef_vec, rng);
		}
		dl_latent(latent_local, global_lev * local_lev.array() / coef_var.array(), coef_vec, rng);
		// prior_alpha_prec = 1 / ((global_lev * local_lev.array() * coef_var.array()).square() * latent_local.array());
		// prior_alpha_prec = (latent_local.array() * coef_var.array().square()) / (global_lev * local_lev.array()).square();
		prior_alpha_prec = latent_local.array() * (coef_var.array() / (global_lev * local_lev.array())).square();
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		dl_dir_griddy(dir_concen, grid_size, local_lev, group_lev[0], rng);
		dl_local_sparsity(local_lev, dir_concen, contem_coef, rng);
		group_lev[0] = dl_global_sparsity(local_lev, dir_concen, contem_coef, rng);
		// dl_latent(latent_local, local_lev, contem_coef, rng);
		dl_latent(latent_local, group_lev[0] * local_lev, contem_coef, rng);
		// prior_chol_prec = 1 / ((group_lev[0] * local_lev.array()).square() * latent_local.array());
		prior_chol_prec =  latent_local.array() / (group_lev[0] * local_lev.array()).square();
	}

	void updateRecords(int id) override {
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	void appendCoefRecords(BVHAR_LIST& list) override {
		list["lambda_record"] = local_record;
		list["tau_record"] = global_record;
	}

	void appendContemRecords(BVHAR_LIST& list) override {
		list["xi_record"] = local_record;
		list["zeta_record"] = group_record;
	}

private:
	double dir_concen, shape, scl;
	int grid_size;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd global_record;
	Eigen::MatrixXd local_record, group_record;
};

/**
 * @brief Generalized Double Pareto (GDP) prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class GdpUpdater : public ShrinkageUpdater {
public:
	GdpUpdater(int num_iter, const GdpParams& params, const GdpInits& inits)
	: ShrinkageUpdater(num_iter, params, inits),
		shape_grid(params._grid_shape), rate_grid(params._grid_rate),
		group_rate(inits._group_rate), group_rate_fac(Eigen::VectorXd::Ones(inits._local.size())),
		gamma_shape(inits._gamma_shape), gamma_rate(inits._gamma_rate),
		local_lev(inits._local),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, local_lev.size())) {}
	virtual ~GdpUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_grp, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BVHAR_BHRNG& rng
	) override {
		gdp_shape_griddy(gamma_shape, gamma_rate, shape_grid, coef_vec, rng);
		gdp_rate_griddy(gamma_rate, gamma_shape, rate_grid, coef_vec, rng);
		gdp_exp_rate(group_rate, gamma_shape, gamma_rate, coef_vec, grp_vec, grp_id, rng);
		for (int j = 0; j < num_grp; ++j) {
			group_rate_fac = (grp_vec.array() == grp_id[j]).select(
				group_rate[j],
				group_rate_fac
			);
		}
		gdp_local_sparsity(local_lev, group_rate_fac, coef_vec, rng);
		prior_alpha_prec = local_lev.array();
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		BVHAR_BHRNG& rng
	) override {
		gdp_shape_griddy(gamma_shape, gamma_rate, shape_grid, contem_coef, rng);
		gdp_rate_griddy(gamma_rate, gamma_shape, rate_grid, contem_coef, rng);
		gdp_exp_rate(group_rate, gamma_shape, gamma_rate, contem_coef, rng);
		gdp_local_sparsity(local_lev, group_rate, contem_coef, rng);
		prior_chol_prec = local_lev.array();
	}

	void updateRecords(int id) override {
		local_record.row(id) = local_lev;
	}

	void appendCoefRecords(BVHAR_LIST& list) override {
		list["lambda_record"] = local_record;
	}

	void appendContemRecords(BVHAR_LIST& list) override {
		list["xi_record"] = local_record;
	}

private:
	int shape_grid, rate_grid;
	Eigen::VectorXd group_rate, group_rate_fac;
	double gamma_shape, gamma_rate;
	Eigen::VectorXd local_lev;
	Eigen::MatrixXd local_record;
};

/**
 * @brief Function to initialize `ShrinkageUpdater`
 * 
 * @tparam UPDATER Shrinkage priors
 * @tparam PARAMS Corresponding parameter struct
 * @tparam INITS Corresponding initialization struct
 * @param num_iter MCMC iteration
 * @param param_prior Shrinkage prior configuration
 * @param param_init Initial values
 * @param prior_type Prior type
 * @return std::unique_ptr<ShrinkageUpdater> 
 */
template <bool isGroup = true>
inline std::unique_ptr<ShrinkageUpdater> initialize_shrinkageupdater(int num_iter, BVHAR_LIST& param_prior, BVHAR_LIST& param_init, int prior_type) {
	std::unique_ptr<ShrinkageUpdater> shrinkage_ptr;
	switch (prior_type) {
		case 1: {
			std::unique_ptr<MinnParams> params_ptr;
			if (BVHAR_CONTAINS(param_prior, "p")) {
				// p is only in coef_prior
				params_ptr = std::make_unique<MinnParams>(param_prior);
			} else {
				// append num_lowerchol to param_prior when contem
				params_ptr = std::make_unique<MinnParams>(param_prior, BVHAR_CAST_INT(param_prior["num"]));
			}
			ShrinkageInits inits(param_init);
			shrinkage_ptr = std::make_unique<MinnUpdater>(num_iter, *params_ptr, inits);
			return shrinkage_ptr;
		}
		case 2: {
			SsvsParams params(param_prior);
			SsvsInits inits(param_init);
			shrinkage_ptr = std::make_unique<SsvsUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 3: {
			ShrinkageParams params(param_prior);
			HorseshoeInits inits(param_init);
			shrinkage_ptr = std::make_unique<HorseshoeUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 4: {
			std::unique_ptr<HierminnParams> params_ptr;
			if (BVHAR_CONTAINS(param_prior, "p")) {
				params_ptr = std::make_unique<HierminnParams>(param_prior);
			} else {
				params_ptr = std::make_unique<HierminnParams>(param_prior, BVHAR_CAST_INT(param_prior["num"]));
			}
			HierminnInits inits(param_init);
			shrinkage_ptr = std::make_unique<HierminnUpdater>(num_iter, *params_ptr, inits);
			return shrinkage_ptr;
		}
		case 5: {
			NgParams params(param_prior);
			NgInits inits(param_init);
			shrinkage_ptr = std::make_unique<NgUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 6: {
			DlParams params(param_prior);
			HorseshoeInits inits(param_init);
			shrinkage_ptr = std::make_unique<DlUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 7: {
			GdpParams params(param_prior);
			GdpInits inits(param_init);
			shrinkage_ptr = std::make_unique<GdpUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
	}
	return shrinkage_ptr;
}

template <bool isGroup = true>
inline std::unique_ptr<ShrinkageUpdater> initialize_shrinkageupdater(
	int num_iter, int num_param, int num_grp, BVHAR_LIST& param_prior, int prior_type,
	BVHAR_BHRNG& rng
) {
	std::unique_ptr<ShrinkageUpdater> shrinkage_ptr;
	switch (prior_type) {
		case 1: {
			std::unique_ptr<MinnParams> params_ptr;
			if (BVHAR_CONTAINS(param_prior, "p")) {
				// p is only in coef_prior
				params_ptr = std::make_unique<MinnParams>(param_prior);
			} else {
				// append num_lowerchol to param_prior when contem
				params_ptr = std::make_unique<MinnParams>(param_prior, num_param);
			}
			ShrinkageInits inits;
			shrinkage_ptr = std::make_unique<MinnUpdater>(num_iter, *params_ptr, inits);
			return shrinkage_ptr;
		}
		case 2: {
			SsvsParams params(param_prior);
			SsvsInits inits(num_param, num_grp, rng);
			shrinkage_ptr = std::make_unique<SsvsUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 3: {
			ShrinkageParams params(param_prior);
			HorseshoeInits inits(num_param, num_grp, rng);
			shrinkage_ptr = std::make_unique<HorseshoeUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 4: {
			std::unique_ptr<HierminnParams> params_ptr;
			if (BVHAR_CONTAINS(param_prior, "p")) {
				params_ptr = std::make_unique<HierminnParams>(param_prior);
			} else {
				params_ptr = std::make_unique<HierminnParams>(param_prior, num_param);
			}
			HierminnInits inits(rng);
			shrinkage_ptr = std::make_unique<HierminnUpdater>(num_iter, *params_ptr, inits);
			return shrinkage_ptr;
		}
		case 5: {
			NgParams params(param_prior);
			NgInits inits(num_param, num_grp, rng);
			shrinkage_ptr = std::make_unique<NgUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 6: {
			DlParams params(param_prior);
			HorseshoeInits inits(num_param, num_grp, rng);
			shrinkage_ptr = std::make_unique<DlUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 7: {
			GdpParams params(param_prior);
			GdpInits inits(num_param, num_grp, rng);
			shrinkage_ptr = std::make_unique<GdpUpdater<isGroup>>(num_iter, params, inits);
			return shrinkage_ptr;
		}
	}
	return shrinkage_ptr;
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H