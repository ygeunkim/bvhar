#ifndef MCMCHS_H
#define MCMCHS_H

#include "bvhardraw.h"
#include "bvharprogress.h"

namespace bvhar {

struct HsParams {
	int _iter;
	Eigen::MatrixXd _x;
	Eigen::MatrixXd _y;
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_group;
	// Eigen::VectorXd _init_global;
	double _init_global;
	double _init_sigma;
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	
	HsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
    const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_group, double init_global,
		const double& init_sigma,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat
	)
	: _iter(num_iter), _x(x), _y(y),
		_init_local(init_local), _init_group(init_group), _init_global(init_global), _init_sigma(init_sigma),
		_grp_id(grp_id), _grp_mat(grp_mat) {}
};

class McmcHs {
public:
	McmcHs(const HsParams& params, unsigned int seed)
	: num_iter(params._iter),
		dim(params._y.cols()), dim_design(params._x.cols()), num_design(params._y.rows()),
		num_coef(dim * dim_design),
		mcmc_step(0), rng(seed),
		design_mat(kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), params._x)),
		response_vec(vectorize_eigen(params._y)),
		lambda_mat(Eigen::MatrixXd::Zero(num_coef, num_coef)),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(vectorize_eigen(grp_mat)), num_grp(grp_id.size()),
		coef_draw(Eigen::VectorXd::Zero(num_coef)), sig_draw(params._init_sigma),
		local_lev(params._init_local), group_lev(params._init_group), global_lev(params._init_global),
		// glob_len(global_lev.size()),
		local_fac(Eigen::VectorXd::Zero(num_coef)),
		shrink_fac(Eigen::VectorXd::Zero(num_coef)),
		latent_local(Eigen::VectorXd::Zero(num_coef)),
		latent_group(Eigen::VectorXd::Zero(num_grp)), latent_global(0.0),
		coef_var(Eigen::VectorXd::Zero(num_coef)),
		coef_var_loc(Eigen::MatrixXd::Zero(dim_design, dim)),
		coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		// global_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		sig_record(Eigen::VectorXd::Zero(num_iter + 1)),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)) {}
	virtual ~McmcHs() = default;
	void addStep() { mcmc_step++; }
	void updateCoefCov() {
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				// global_lev.segment(j, 1).replicate(dim_design, dim),
				group_lev.segment(j, 1).replicate(dim_design, dim),
				coef_var_loc
			);
		}
		coef_var = vectorize_eigen(coef_var_loc);
		local_fac.array() = coef_var.array() * local_lev.array();
		// build_shrink_mat(lambda_mat, coef_var, local_lev);
		lambda_mat.setZero();
		lambda_mat.diagonal() = 1 / (global_lev * local_fac.array());
		shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
	}
	virtual void updateCoef() {
		horseshoe_coef(coef_draw, response_vec, design_mat, sig_draw, lambda_mat, rng);
		// sig_draw = horseshoe_var(response_vec, design_mat, lambda_mat, rng);
		sig_draw = horseshoe_var(response_vec, design_mat, coef_draw, lambda_mat, rng);
	}
	void updateCov() {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_latent(latent_global, global_lev, rng);
		global_lev = horseshoe_global_sparsity(latent_global, local_fac, coef_draw, 1, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_draw, sig_draw, rng);
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_draw, sig_draw, rng);
		// horseshoe_mn_global_sparsity(global_lev, grp_vec, grp_id, latent_global, local_lev, coef_draw, sig_draw, rng);
	}
	virtual void updateRecords() {
		shrink_record.row(mcmc_step) = shrink_fac;
		coef_record.row(mcmc_step) = coef_draw;
		sig_record[mcmc_step] = sig_draw;
		local_record.row(mcmc_step) = local_lev.cwiseSqrt();
		group_record.row(mcmc_step) = group_lev.cwiseSqrt();
		global_record[mcmc_step] = sqrt(global_lev);
	}
	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefCov();
		updateCoef();
		updateCov();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = coef_record,
			Rcpp::Named("lambda_record") = local_record,
			Rcpp::Named("eta_record") = group_record,
			Rcpp::Named("tau_record") = global_record,
			Rcpp::Named("sigma_record") = sig_record,
			Rcpp::Named("kappa_record") = shrink_record
		);
		for (auto& record : res) {
			if (Rcpp::is<Rcpp::NumericMatrix>(record)) {
				record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
			} else {
				record = thin_record(Rcpp::as<Eigen::VectorXd>(record), num_iter, num_burn, thin);
			}
		}
		return res;
	}
protected:
	int num_iter;
	int dim; // k
	int dim_design; // kp(+1)
	int num_design; // n = T - p
	int num_coef;
	std::mutex mtx;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::MatrixXd design_mat;
	Eigen::VectorXd response_vec;
	Eigen::MatrixXd lambda_mat; // covariance
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	Eigen::VectorXd coef_draw;
	double sig_draw;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	// int glob_len;
	Eigen::VectorXd local_fac;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::MatrixXd coef_record;
  Eigen::MatrixXd local_record;
  // Eigen::MatrixXd global_record; // tau1: own-lag, tau2: cross-lag, ...
	Eigen::MatrixXd group_record; // eta1: own-lag, eta2: cross-lag, ...
	Eigen::VectorXd global_record; // tau
  Eigen::VectorXd sig_record;
  Eigen::MatrixXd shrink_record;
};

class BlockHs : public McmcHs {
public:
	BlockHs(const HsParams& params, unsigned int seed) : McmcHs(params, seed), block_coef(Eigen::VectorXd::Zero(num_coef + 1)) {}
	virtual ~BlockHs() = default;
	void updateCoef() override { horseshoe_coef_var(block_coef, response_vec, design_mat, lambda_mat, rng); }
	void updateRecords() override {
		shrink_record.row(mcmc_step) = shrink_fac;
		coef_record.row(mcmc_step) = block_coef.tail(num_coef);
		sig_record[mcmc_step] = block_coef[0];
		local_record.row(mcmc_step) = local_lev.cwiseSqrt();
		group_record.row(mcmc_step) = group_lev.cwiseSqrt();
		global_record[mcmc_step] = sqrt(global_lev);
	}
private:
	Eigen::VectorXd block_coef;
};

class FastHs : public McmcHs {
public:
	FastHs(const HsParams& params, unsigned int seed) : McmcHs(params, seed) {}
	virtual ~FastHs() = default;
	void updateCoef() override {
		horseshoe_fast_coef(
			coef_draw,
			response_vec / sqrt(sig_draw),
			design_mat / sqrt(sig_draw),
			sig_draw * lambda_mat,
			rng
		);
		// sig_draw = horseshoe_var(response_vec, design_mat, lambda_mat, rng);
		sig_draw = horseshoe_var(response_vec, design_mat, coef_draw, lambda_mat, rng);
	}
	void updateRecords() override {
		shrink_record.row(mcmc_step) = shrink_fac;
		coef_record.row(mcmc_step) = coef_draw;
		sig_record[mcmc_step] = sig_draw;
		local_record.row(mcmc_step) = local_lev.cwiseSqrt();
		group_record.row(mcmc_step) = group_lev.cwiseSqrt();
		global_record[mcmc_step] = sqrt(global_lev);
	}
};

} // namespace bvhar

#endif // MCMCHS_H