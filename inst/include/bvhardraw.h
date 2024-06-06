#ifndef BVHARDRAW_H
#define BVHARDRAW_H

#include "bvharsim.h"
#include <set>

namespace bvhar {

struct RegParams {
	int _iter;
	Eigen::MatrixXd _x;
	Eigen::MatrixXd _y;
	Eigen::VectorXd _sig_shp;
	Eigen::VectorXd _sig_scl;
	Eigen::VectorXd _mean_non;
	double _sd_non;
	bool _mean;

	RegParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& spec, Rcpp::List& intercept,
		bool include_mean
	)
	: _iter(num_iter), _x(x), _y(y),
		_sig_shp(Rcpp::as<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(Rcpp::as<Eigen::VectorXd>(spec["scale"])),
		_mean_non(Rcpp::as<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(intercept["sd_non"]), _mean(include_mean) {}
};

struct RegInits {
	Eigen::MatrixXd _coef;
	Eigen::VectorXd _contem;

	RegInits(const RegParams& params) {
		_coef = (params._x.transpose() * params._x).llt().solve(params._x.transpose() * params._y); // OLS
		int dim = params._y.cols();
		int num_lowerchol = dim * (dim - 1) / 2;
		_contem = .001 * Eigen::VectorXd::Zero(num_lowerchol);
	}
	RegInits(Rcpp::List& init)
	: _coef(Rcpp::as<Eigen::MatrixXd>(init["init_coef"])),
		_contem(Rcpp::as<Eigen::VectorXd>(init["init_contem"])) {}
};

struct RegRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)

	RegRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	RegRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}
	// void assignRecords(
	// 	int id,
	// 	const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	// ) {
	// 	coef_record.row(id) = coef_vec;
	// 	contem_coef_record.row(id) = contem_coef;
	// 	fac_record.row(id) = diag_vec;
	// }

	Eigen::VectorXd computeActivity(double level) {
		Eigen::VectorXd lower_ci(coef_record.cols());
		Eigen::VectorXd upper_ci(coef_record.cols());
		Eigen::VectorXd selection(coef_record.cols());
		for (int i = 0; i < coef_record.cols(); ++i) {
			// lower_ci[i] = quantile_lower(coef_record.col(i), level / 2);
			// upper_ci[i] = quantile_upper(coef_record.col(i), 1 - level / 2);
			selection[i] = quantile_lower(coef_record.col(i), level / 2) * quantile_upper(coef_record.col(i), 1 - level / 2) < 0 ? 0.0 : 1.1;
		}
		return selection;
	}
};

struct SsvsRecords {
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd contem_dummy_record;
	Eigen::MatrixXd contem_weight_record;

	SsvsRecords() : coef_dummy_record(), coef_weight_record(), contem_dummy_record(), contem_weight_record() {}
	SsvsRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: coef_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_alpha)),
		coef_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		contem_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_lowerchol)),
		contem_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	SsvsRecords(
		const Eigen::MatrixXd& coef_dummy_record, const Eigen::MatrixXd& coef_weight_record,
		const Eigen::MatrixXd& contem_dummy_record, const Eigen::MatrixXd& contem_weight_record
	)
	: coef_dummy_record(coef_dummy_record), coef_weight_record(coef_weight_record),
		contem_dummy_record(contem_dummy_record), contem_weight_record(contem_weight_record) {}
	void assignRecords(int id, const Eigen::VectorXd& coef_dummy, const Eigen::VectorXd& coef_weight, const Eigen::VectorXd& contem_dummy, const Eigen::VectorXd& contem_weight) {
		coef_dummy_record.row(id) = coef_dummy;
		coef_weight_record.row(id) = coef_weight;
		contem_dummy_record.row(id) = contem_dummy;
		contem_weight_record.row(id) = contem_weight;
	}
};

struct HorseshoeRecords {
	Eigen::MatrixXd local_record;
	Eigen::MatrixXd group_record;
	// Eigen::MatrixXd global_record;
	Eigen::VectorXd global_record;
	Eigen::MatrixXd shrink_record;

	HorseshoeRecords() : local_record(), global_record(), shrink_record() {}
	HorseshoeRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		// global_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)) {}
	// HorseshoeRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& global_record, const Eigen::MatrixXd& shrink_record)
	HorseshoeRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& group_record, const Eigen::VectorXd& global_record, const Eigen::MatrixXd& shrink_record)
	: local_record(local_record), group_record(group_record), global_record(global_record), shrink_record(shrink_record) {}
	// void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& global_lev) {
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		shrink_record.row(id) = shrink_fac;
		local_record.row(id) = local_lev;
		// global_record.row(id) = global_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}
};

// Numerically Stable Log Marginal Likelihood Excluding Constant Term
// 
// This function computes log of ML stable,
// excluding the constant term.
// 
// @param dim Dimension of the time series
// @param num_design The number of the data matrix, \eqn{n = T - p}
// @param prior_prec Prior precision of Matrix Normal distribution
// @param prior_scale Prior scale of Inverse-Wishart distribution
// @param mn_prec Posterior precision of Matrix Normal distribution
// @param iw_scale Posterior scale of Inverse-Wishart distribution
// @param posterior_shape Posterior shape of Inverse-Wishart distribution
inline double compute_logml(int dim, int num_design, Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale,
														Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale, int posterior_shape) {
  Eigen::LLT<Eigen::MatrixXd> lltOfmn(prior_prec.inverse());
  Eigen::MatrixXd chol_mn = lltOfmn.matrixL();
  Eigen::MatrixXd stable_mat_a = chol_mn.transpose() * (mn_prec - prior_prec) * chol_mn;
  Eigen::LLT<Eigen::MatrixXd> lltOfiw(prior_scale.inverse());
  Eigen::MatrixXd chol_iw = lltOfiw.matrixL();
  Eigen::MatrixXd stable_mat_b = chol_iw.transpose() * (iw_scale - prior_scale) * chol_iw;
  // eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(stable_mat_a);
  Eigen::VectorXd a_eigen = es_a.eigenvalues();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_b(stable_mat_b);
  Eigen::VectorXd b_eigen = es_b.eigenvalues();
  // sum of log(1 + eigenvalues)
  double a_term = a_eigen.array().log1p().sum();
  double b_term = b_eigen.array().log1p().sum();
  // result
  return - num_design / 2.0 * log(prior_scale.determinant()) - dim / 2.0 * a_term - posterior_shape / 2.0 * b_term;
}

// Log of Joint Posterior Density of Hyperparameters
// 
// This function computes the log of joint posterior density of hyperparameters.
// 
// @param cand_gamma Candidate value of hyperparameters following Gamma distribution
// @param cand_invgam Candidate value of hyperparameters following Inverse Gamma distribution
// @param dim Dimension of the time series
// @param num_design The number of the data matrix, \eqn{n = T - p}
// @param prior_prec Prior precision of Matrix Normal distribution
// @param prior_scale Prior scale of Inverse-Wishart distribution
// @param mn_prec Posterior precision of Matrix Normal distribution
// @param iw_scale Posterior scale of Inverse-Wishart distribution
// @param posterior_shape Posterior shape of Inverse-Wishart distribution
// @param gamma_shape Shape of hyperprior Gamma distribution
// @param gamma_rate Rate of hyperprior Gamma distribution
// @param invgam_shape Shape of hyperprior Inverse gamma distribution
// @param invgam_scl Scale of hyperprior Inverse gamma distribution
inline double jointdens_hyperparam(double cand_gamma, Eigen::VectorXd cand_invgam, int dim, int num_design,
                            			 Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, int prior_shape,
                            			 Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale,
                            			 int posterior_shape, double gamma_shp, double gamma_rate, double invgam_shp, double invgam_scl) {
  double res = compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_shape);
  res += -dim * num_design / 2.0 * log(M_PI) +
    lmgammafn((prior_shape + num_design) / 2.0, dim) -
    lmgammafn(prior_shape / 2.0, dim); // constant term
  res += gamma_dens(cand_gamma, gamma_shp, 1 / gamma_rate, true); // gamma distribution
  for (int i = 0; i < cand_invgam.size(); i++) {
    res += invgamma_dens(cand_invgam[i], invgam_shp, invgam_scl, true); // inverse gamma distribution
  }
  return res;
}

// Building Spike-and-slab SD Diagonal Matrix
// 
// In MCMC process of SSVS, this function computes diagonal matrix \eqn{D} or \eqn{D_j} defined by spike-and-slab sd.
// 
// @param spike_sd Standard deviance for Spike normal distribution
// @param slab_sd Standard deviance for Slab normal distribution
// @param mixture_dummy Indicator vector (0-1) corresponding to each element
inline Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy) {
  Eigen::VectorXd res(spike_sd.size());
  res.array() = (1 - mixture_dummy.array()) * spike_sd.array() + mixture_dummy.array() * slab_sd.array(); // diagonal term = spike_sd if mixture_dummy = 0 while slab_sd if mixture_dummy = 1
  return res;
}

// Building Lower Triangular Matrix
// 
// In MCMC, this function builds \eqn{L} given \eqn{a} vector.
// 
// @param dim Dimension (dim x dim) of L
// @param lower_vec Vector a
inline Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Identity(dim, dim);
  int id = 0;
  for (int i = 1; i < dim; i++) {
    res.row(i).segment(0, i) = lower_vec.segment(id, i);
    id += i;
  }
  return res;
}

// Generating the Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates the diagonal component \eqn{\Psi} from variance matrix
// 
// @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
// @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
// @param shape Gamma shape parameters for precision matrix
// @param rate Gamma rate parameters for precision matrix
// @param num_design The number of sample used, \eqn{n = T - p}
inline void ssvs_chol_diag(Eigen::VectorXd& chol_diag, Eigen::MatrixXd& sse_mat, Eigen::VectorXd& DRD,
													 Eigen::VectorXd& shape, Eigen::VectorXd& rate, int num_design, boost::mt19937& rng) {
  int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  shape.array() += (double)num_design / 2;
  rate[0] += sse_mat(0, 0) / 2;
  chol_diag[0] = sqrt(gamma_rand(shape[0], 1 / rate[0], rng)); // psi[11]^2 ~ Gamma(shape, rate)
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1); // (s1j, ..., sj-1,j)
    rate[j] += (
      sse_mat(j, j) - 
        sse_colvec.segment(0, j).transpose() * 
        (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j)) * 
        sse_colvec.segment(0, j)
    ) / 2;
    chol_diag[j] = sqrt(gamma_rand(shape[j], 1 / rate[j], rng)); // psi[jj]^2 ~ Gamma(shape, rate)
    block_id += j;
  }
}

// Generating the Off-Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates the off-diagonal component \eqn{\Psi} of variance matrix
// 
// @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
// @param chol_diag Diagonal element of the cholesky factor
// @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
inline void ssvs_chol_off(Eigen::VectorXd& chol_off, Eigen::MatrixXd& sse_mat,
													Eigen::VectorXd& chol_diag, Eigen::VectorXd& DRD, boost::random::mt19937& rng) {
	int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::MatrixXd normal_variance(dim - 1, dim - 1);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  Eigen::VectorXd normal_mean(dim - 1);
  // Eigen::VectorXd res(num_param);
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1);
    normal_variance.topLeftCorner(j, j) = (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j));
    normal_mean.segment(0, j) = -chol_diag[j] * normal_variance.topLeftCorner(j, j) * sse_colvec.segment(0, j);
    chol_off.segment(block_id, j) = vectorize_eigen(sim_mgaussian_chol(1, normal_mean.segment(0, j), normal_variance.topLeftCorner(j, j), rng));
    block_id += j;
  }
}

// Filling Cholesky Factor Upper Triangular Matrix
// 
// This function builds a cholesky factor matrix \eqn{\Psi} (upper triangular) using diagonal component vector and off-diagonal component vector.
// 
// @param diag_vec Diagonal components
// @param off_diagvec Off-diagonal components
inline Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec; // psi
  int id = 0; // length of eta = m(m-1)/2
  // assign eta_j = (psi_1j, ..., psi_j-1,j)
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i]; // assign i-th row = psi_ij
    }
    id += j;
  }
  return res;
}

inline Eigen::MatrixXd build_cov(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec;
  int id = 0;
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i]; // assign i-th row = psi_ij
      res(j, i) = res(i, j);
    }
    id += j;
  }
  return res;
}

// Generating Coefficient Vector in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{\alpha_j} conditional posterior.
// 
// @param prior_mean The prior mean vector of the VAR coefficient vector
// @param prior_sd Diagonal prior sd matrix of the VAR coefficient vector
// @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
// @param coef_ols OLS (MLE) estimator of the VAR coefficient
// @param chol_factor Cholesky factor of variance matrix
inline void ssvs_coef(Eigen::VectorXd& coef, Eigen::VectorXd& prior_mean, Eigen::VectorXd& prior_sd,
											Eigen::MatrixXd& XtX, Eigen::VectorXd& coef_ols,
											Eigen::MatrixXd& chol_factor, boost::random::mt19937& rng) {
	int num_coef = prior_sd.size();
  Eigen::MatrixXd scaled_xtx = kronecker_eigen((chol_factor * chol_factor.transpose()).eval(), XtX); // Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd prior_prec = Eigen::MatrixXd::Zero(num_coef, num_coef);
  prior_prec.diagonal() = 1 / prior_sd.array().square();
  // Eigen::MatrixXd normal_variance = (scaled_xtx + prior_prec).llt().solve(Eigen::MatrixXd::Identity(num_coef, num_coef)); // Delta
  // Eigen::VectorXd normal_mean = normal_variance * (scaled_xtx * coef_ols + prior_prec * prior_mean); // mu
  // coef = vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance, rng));
	Eigen::VectorXd standard_normal(num_coef);
	for (int i = 0; i < num_coef; i++) {
		standard_normal[i] = normal_rand(rng);
	}
	Eigen::MatrixXd normal_variance = scaled_xtx + prior_prec; // Delta^(-1)
	Eigen::LLT<Eigen::MatrixXd> llt_sig(normal_variance);
	Eigen::MatrixXd normal_mean = llt_sig.solve(scaled_xtx * coef_ols + prior_prec * prior_mean);
	coef = normal_mean + llt_sig.matrixU().solve(standard_normal);
}

// Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
// 
// @param param_obs Realized parameters vector
// @param sd_numer Standard deviance for Slab normal distribution, which will be used for numerator.
// @param sd_denom Standard deviance for Spike normal distribution, which will be used for denominator.
// @param slab_weight Proportion of nonzero coefficients
inline void ssvs_dummy(
	Eigen::VectorXd& dummy, Eigen::VectorXd param_obs,
	Eigen::VectorXd& sd_numer, Eigen::VectorXd& sd_denom, Eigen::VectorXd& slab_weight,
	boost::random::mt19937& rng
) {
  int num_latent = slab_weight.size();
	Eigen::VectorXd exp_u1 = -param_obs.array().square() / (2 * sd_numer.array().square());
	Eigen::VectorXd exp_u2 = -param_obs.array().square() / (2 * sd_denom.array().square());
	Eigen::VectorXd max_exp = exp_u1.cwiseMax(exp_u2); // use log-sum-exp against overflow
	exp_u1 = slab_weight.array() * (exp_u1 - max_exp).array().exp() / sd_numer.array();
	exp_u2 = (1 - slab_weight.array()) * (exp_u2 - max_exp).array().exp() / sd_denom.array();
  for (int i = 0; i < num_latent; i++) {
		dummy[i] = ber_rand(exp_u1[i] / (exp_u1[i] + exp_u2[i]), rng);
  }
}

// Generating Slab Weight Vector in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{p_j}.
// 
// @param param_obs Indicator variables
// @param prior_s1 First prior shape of Beta distribution
// @param prior_s2 Second prior shape of Beta distribution
inline void ssvs_weight(Eigen::VectorXd& weight, Eigen::VectorXd param_obs, double prior_s1, double prior_s2, boost::random::mt19937& rng) {
  int num_latent = param_obs.size();
  double post_s1 = prior_s1 + param_obs.sum(); // s1 + number of ones
  double post_s2 = prior_s2 + num_latent - param_obs.sum(); // s2 + number of zeros
  for (int i = 0; i < num_latent; i++) {
		weight[i] = beta_rand(post_s1, post_s2, rng);
  }
}

// Generating Slab Weight Vector in MN-SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{p_j}.
// 
// @param grp_vec Group vector
// @param grp_id Unique group id
// @param param_obs Indicator variables
// @param prior_s1 First prior shape of Beta distribution
// @param prior_s2 Second prior shape of Beta distribution
inline void ssvs_mn_weight(Eigen::VectorXd& weight, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
  												 Eigen::VectorXd& param_obs, Eigen::VectorXd& prior_s1, Eigen::VectorXd& prior_s2, boost::random::mt19937& rng) {
  int num_grp = grp_id.size();
  int num_latent = param_obs.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		global_id = grp_vec.array() == grp_id[i];
		mn_size = global_id.count();
    Eigen::VectorXd mn_param(mn_size);
		for (int j = 0, k = 0; j < num_latent; ++j) {
			if (global_id[j]) {
				mn_param[k++] = param_obs[j];
			}
		}
    weight[i] = beta_rand(prior_s1[i] + mn_param.sum(), prior_s2[i] + mn_size - mn_param.sum(), rng);
  }
}

// Generating the Equation-wise Coefficients Vector and Contemporaneous Coefficients
// 
// This function generates j-th column of coefficients matrix and j-th row of impact matrix using precision sampler.
//
// @param x Design matrix of the system
// @param y Response vector of the system
// @param prior_mean Prior mean vector
// @param prior_prec Prior precision matrix
// @param innov_prec Stacked precision matrix of innovation
inline void varsv_regression(Eigen::Ref<Eigen::VectorXd> coef, Eigen::MatrixXd& x, Eigen::VectorXd& y,
														 Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec, boost::random::mt19937& rng) {
  int dim = prior_mean.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
		res[i] = normal_rand(rng);
  }
  Eigen::MatrixXd post_sig = prior_prec + x.transpose() * x;
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(post_sig);
  Eigen::VectorXd post_mean = lltOfscale.solve(prior_prec * prior_mean + x.transpose() * y);
	coef = post_mean + lltOfscale.matrixU().solve(res);
}

// Generating log-volatilities in MCMC
// 
// In MCMC, this function samples log-volatilities \eqn{h_{it}} vector using auxiliary mixture sampling
// 
// @param sv_vec log-volatilities vector
// @param init_sv Initial log-volatility
// @param sv_sig Variance of log-volatilities
// @param latent_vec Auxiliary residual vector
inline void varsv_ht(Eigen::Ref<Eigen::VectorXd> sv_vec, double init_sv,
										 double sv_sig, Eigen::Ref<Eigen::VectorXd> latent_vec, boost::random::mt19937& rng) {
  int num_design = sv_vec.size(); // h_i1, ..., h_in for i = 1, .., k
  // 7-component normal mixutre
  Eigen::VectorXd pj(7); // p_t
  pj << 0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575;
  Eigen::VectorXd muj(7); // mu_t
  muj << -10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819;
  muj.array() -= 1.2704;
  Eigen::VectorXd sigj(7); // sig_t^2
  sigj << 5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261;
  Eigen::VectorXd sdj = sigj.cwiseSqrt();
  Eigen::VectorXi binom_latent(num_design);
  Eigen::VectorXd ds(num_design); // (mu_st - 1.2704)
  Eigen::MatrixXd inv_sig_s = Eigen::MatrixXd::Zero(num_design, num_design); // diag(1 / sig_st^2)
  Eigen::VectorXd inv_method(num_design); // inverse transform method
  Eigen::MatrixXd mixture_pdf(num_design, 7);
  Eigen::MatrixXd mixture_cumsum = Eigen::MatrixXd::Zero(num_design, 7);
  for (int i = 0; i < num_design; i++) {
		inv_method[i] = unif_rand(0, 1, rng);
  }
  for (int i = 0; i < 7; i++) {
    mixture_pdf.col(i) = (-((latent_vec.array() - sv_vec.array() - muj[i]).array() / sdj[i]).array().square() / 2).exp() * pj[i] / (sdj[i] * sqrt(2 * M_PI));
  }
  mixture_pdf.array().colwise() /= mixture_pdf.rowwise().sum().array();
  for (int i = 0; i < 7; i++) {
    mixture_cumsum.block(0, i, num_design, 7 - i) += mixture_pdf.col(i).rowwise().replicate(7 - i);
  }
  binom_latent.array() = 7 - (inv_method.rowwise().replicate(7).array() < mixture_cumsum.array()).cast<int>().rowwise().sum().array(); // 0 to 6 for indexing
  Eigen::MatrixXd diff_mat = Eigen::MatrixXd::Identity(num_design, num_design);
  for (int i = 0; i < num_design - 1; i++) {
    ds[i] = muj[binom_latent[i]];
    inv_sig_s(i, i) = 1 / sigj[binom_latent[i]];
    diff_mat(i + 1, i) = -1;
  }
  ds[num_design - 1] = muj[binom_latent[num_design - 1]];
  inv_sig_s(num_design - 1, num_design - 1) = 1 / sigj[binom_latent[num_design - 1]];
  Eigen::MatrixXd HtH = diff_mat.transpose() * diff_mat;
  Eigen::VectorXd res(num_design);
  for (int i = 0; i < num_design; i++) {
		res[i] = normal_rand(rng);
  }
  Eigen::MatrixXd post_sig = HtH / sv_sig + inv_sig_s;
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(post_sig);
  Eigen::VectorXd post_mean = lltOfscale.solve(
    HtH * init_sv * Eigen::VectorXd::Ones(num_design) / sv_sig + inv_sig_s * (latent_vec - ds)
  );
	sv_vec = post_mean + lltOfscale.matrixU().solve(res);
}

// Generating sig_h in MCMC
// 
// In MCMC, this function samples \eqn{\sigma_h^2} in VAR-SV.
// 
// @param shp Prior shape of sigma
// @param scl Prior scale of sigma
// @param init_sv Initial log volatility
// @param h1 Time-varying h1 matrix
inline void varsv_sigh(Eigen::VectorXd& sv_sig, Eigen::VectorXd& shp, Eigen::VectorXd& scl,
											 Eigen::VectorXd& init_sv, Eigen::MatrixXd& h1, boost::random::mt19937& rng) {
  int dim = init_sv.size();
  int num_design = h1.rows();
  Eigen::MatrixXd h_slide(num_design, dim); // h_ij, j = 0, ..., n - 1
  h_slide.row(0) = init_sv;
  h_slide.bottomRows(num_design - 1) = h1.topRows(num_design - 1);
  for (int i = 0; i < dim; i++) {
    sv_sig[i] = 1 / gamma_rand(
      shp[i] + num_design / 2,
			1 / (scl[i] + (h1.array() - h_slide.array()).square().sum() / 2),
			rng
    );
  }
}

// Generating h0 in MCMC
// 
// In MCMC, this function samples h0 in VAR-SV.
// 
// @param prior_mean Prior mean vector of h0.
// @param prior_prec Prior precision matrix of h0.
// @param init_sv Initial log volatility
// @param h1 h1
// @param sv_sig Variance of log volatility
inline void varsv_h0(Eigen::VectorXd& h0, Eigen::VectorXd& prior_mean, Eigen::MatrixXd& prior_prec,
              			 Eigen::VectorXd h1, Eigen::VectorXd& sv_sig, boost::random::mt19937& rng) {
  int dim = h1.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
		res[i] = normal_rand(rng);
  }
  Eigen::MatrixXd post_h0_prec(dim, dim); // k_h0
  Eigen::MatrixXd h_diagprec = Eigen::MatrixXd::Zero(dim, dim); // diag(1 / sigma_h^2)
  h_diagprec.diagonal() = 1 / sv_sig.array();
  Eigen::MatrixXd post_h0_sig = prior_prec + h_diagprec;
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(post_h0_sig);
  Eigen::VectorXd post_mean = lltOfscale.solve(prior_prec * prior_mean + h_diagprec * h1);
	h0 = post_mean + lltOfscale.matrixU().solve(res);
}

// Building a Inverse Diagonal Matrix by Global and Local Hyperparameters
// 
// In MCMC process of Horseshoe, this function computes diagonal matrix \eqn{\Lambda_\ast^{-1}} defined by
// global and local sparsity levels.
// 
// @param global_hyperparam Global sparsity hyperparameters
// @param local_hyperparam Local sparsity hyperparameters
inline void build_shrink_mat(Eigen::MatrixXd& cov, Eigen::VectorXd& global_hyperparam, Eigen::Ref<Eigen::VectorXd> local_hyperparam) {
	cov.setZero();
  cov.diagonal() = 1 / (local_hyperparam.array() * global_hyperparam.array()).square();
	// cov.diagonal() = 1 / (local_hyperparam.array() * global_hyperparam.array());
	// cov.diagonal() = -2 * (local_hyperparam.array() * global_hyperparam.array()).log();
	// cov.diagonal() = (-2 * (local_hyperparam.array() * global_hyperparam.array()).log()).exp();
	// cov.diagonal() = cov.diagonal().array().exp();
}

// Generating the Coefficient Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_coef(Eigen::VectorXd& coef, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat,
                    			 double var, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng) {
	int dim = coef.size();
	Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
		res[i] = normal_rand(rng);
  }
	Eigen::MatrixXd post_sig = shrink_mat / var + design_mat.transpose() * design_mat;
	Eigen::LLT<Eigen::MatrixXd> llt_sig(post_sig);
	Eigen::VectorXd post_mean = llt_sig.solve(design_mat.transpose() * response_vec);
	// coef = post_mean + llt_sig.matrixU().solve(design_mat.transpose() * response_vec);
	coef = post_mean + llt_sig.matrixU().solve(res);
}

// Generating the Coefficient Vector using Fast Sampling
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_fast_coef(Eigen::VectorXd& coef, Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat,
												 				Eigen::MatrixXd shrink_mat, boost::random::mt19937& rng) {
  int num_coef = design_mat.cols(); // k^2 kp(+1)
  int num_sur = response_vec.size(); // nk-dim
  Eigen::MatrixXd sur_identity = Eigen::MatrixXd::Identity(num_sur, num_sur);
  Eigen::VectorXd u_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_coef), shrink_mat, rng));
  Eigen::VectorXd delta_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_sur), sur_identity, rng));
  Eigen::VectorXd nu = design_mat * u_vec + delta_vec;
  Eigen::VectorXd lin_solve = (design_mat * shrink_mat * design_mat.transpose() + sur_identity).llt().solve(
    response_vec - nu
  );
  coef = u_vec + shrink_mat * design_mat.transpose() * lin_solve;
}

// Generating the Coefficient Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_coef_var(Eigen::VectorXd& coef_var, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat,
															 Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng) {
  int dim = design_mat.cols();
  int sample_size = response_vec.size();
  Eigen::MatrixXd prec_mat = (design_mat.transpose() * design_mat + shrink_mat).llt().solve(
    Eigen::MatrixXd::Identity(dim, dim)
  );
  double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * prec_mat * design_mat.transpose()) * response_vec;
  coef_var[0] = 1 / gamma_rand(sample_size / 2, scl / 2, rng);
  coef_var.tail(dim) = vectorize_eigen(
    sim_mgaussian_chol(1, prec_mat * design_mat.transpose() * response_vec, coef_var[0] * prec_mat, rng)
  );
}

// Generating the Prior Variance Constant in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the prior variance.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param coef_vec Coefficients vector
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
// inline double horseshoe_var(Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng) {
//   int sample_size = response_vec.size();
//   double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * shrink_mat * design_mat.transpose()) * response_vec;
//   return 1 / gamma_rand(sample_size / 2, 2 / scl, rng);
// }

inline double horseshoe_var(Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::VectorXd& coef_vec, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng) {
  return 1 / gamma_rand(
		design_mat.size() / 2,
		2 / ((response_vec - design_mat * coef_vec).squaredNorm() + coef_vec.transpose() * shrink_mat * coef_vec), rng
	);
}

// Generating the Squared Grouped Local Sparsity Hyperparameters Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the local sparsity hyperparameters vector.
// 
// @param local_latent Latent vectors defined for local sparsity vector
// @param global_hyperparam Squared global sparsity hyperparameter vector
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline void horseshoe_local_sparsity(Eigen::VectorXd& local_lev, Eigen::VectorXd& local_latent, Eigen::VectorXd& global_hyperparam,
                            				 Eigen::Ref<Eigen::VectorXd> coef_vec, double prior_var, boost::random::mt19937& rng) {
  int dim = coef_vec.size();
	Eigen::VectorXd invgam_scl = (1 / local_latent.array() + coef_vec.array().square() / (2 * prior_var * global_hyperparam.array().square())).cwiseInverse();
  for (int i = 0; i < dim; i++) {
		local_lev[i] = sqrt(1 / gamma_rand(1.0, invgam_scl[i], rng));
  }
}

// Generating the Squared Grouped Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the grouped global sparsity hyperparameter.
// 
// @param global_latent Latent global vector
// @param local_hyperparam Squared local sparsity hyperparameters vector
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline double horseshoe_global_sparsity(double global_latent, Eigen::Ref<Eigen::VectorXd> local_hyperparam,
                                 				Eigen::Ref<Eigen::VectorXd> coef_vec, double prior_var, boost::random::mt19937& rng) {
  int dim = coef_vec.size();
	// double invgam_scl = 1 / global_latent + (coef_vec.array().square() / (2 * prior_var * local_hyperparam.array().square())).sum();
	return sqrt(
		1 / gamma_rand(
			(dim + 1) / 2,
			1 / (1 / global_latent + (coef_vec.array().square() / (2 * prior_var * local_hyperparam.array().square())).sum()),
			rng
		)
	);
}

// Generating the Squared Grouped Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the grouped global sparsity hyperparameter.
// 
// @param glob_lev Squared global sparsity hyperparameters
// @param grp_vec Group vector
// @param grp_id Unique group id
// @param global_latent Latent global vector
// @param local_hyperparam Squared local sparsity hyperparameters
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline void horseshoe_mn_global_sparsity(Eigen::VectorXd& global_lev, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
                                  			 Eigen::VectorXd& global_latent, Eigen::VectorXd& local_hyperparam,
																				 Eigen::Ref<Eigen::VectorXd> coef_vec, double prior_var, boost::random::mt19937& rng) {
  int num_grp = grp_id.size();
  int num_coef = coef_vec.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		global_id = grp_vec.array() == grp_id[i];
		mn_size = global_id.count();
    Eigen::VectorXd mn_coef(mn_size);
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (global_id[j]) {
				mn_coef[k] = coef_vec[j];
				mn_local[k++] = local_hyperparam[j];
			}
		}
    global_lev[i] = horseshoe_global_sparsity(global_latent[i], mn_local, mn_coef, prior_var, rng); 
  }
}

// For group shrinkage
inline void horseshoe_mn_sparsity(Eigen::VectorXd& group_lev, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
                                  Eigen::VectorXd& group_latent, double& global_lev, Eigen::VectorXd& local_hyperparam,
																	Eigen::Ref<Eigen::VectorXd> coef_vec, double prior_var, boost::random::mt19937& rng) {
  int num_grp = grp_id.size();
  int num_coef = coef_vec.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    Eigen::VectorXd mn_coef(mn_size);
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (group_id[j]) {
				mn_coef[k] = coef_vec[j];
				mn_local[k++] = global_lev * local_hyperparam[j];
			}
		}
    group_lev[i] = horseshoe_global_sparsity(group_latent[i], mn_local, mn_coef, prior_var, rng); 
  }
}

// Generating the Latent Vector for Sparsity Hyperparameters in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the latent vector for local sparsity hyperparameters.
// 
// @param hyperparam sparsity hyperparameters vector
inline void horseshoe_latent(Eigen::VectorXd& latent, Eigen::VectorXd& hyperparam, boost::random::mt19937& rng) {
  int dim = hyperparam.size();
  for (int i = 0; i < dim; i++) {
		latent[i] = 1 / gamma_rand(1.0, 1 / (1 + 1 / (hyperparam[i] * hyperparam[i])), rng);
  }
}
// overloading
inline void horseshoe_latent(double& latent, double& hyperparam, boost::random::mt19937& rng) {
  latent = 1 / gamma_rand(1.0, 1 / (1 + 1 / (hyperparam * hyperparam)), rng);
}

// Generating lambda of Minnesota-SV
// 
// @param lambda lambda1 or lambda2
// @param shape Gamma prior shape
// @param rate Gamma prior rate
// @param coef_vec Coefficients vector
// @param coef_mean Prior mean of coefficients vector
// @param coef_prec Prior precision matrix, sigma_i / sigma_j
// @param grp_vec Group vector
// @param grp_id Unique group id only for own-lag or cross-lag
// @param grp_vec Group vector
// @param lag_id id for own-lag or cross-lag
// @param rng boost rng
inline void minnesota_lambda(double& lambda, double& shape, double& rate, Eigen::Ref<Eigen::VectorXd> coef,
														 Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::MatrixXd& coef_prec,
														 Eigen::VectorXi& grp_vec, std::set<int>& grp_id, boost::random::mt19937& rng) {
	int num_alpha = coef.size();
	int mn_size = 0;
	double gig_chi = 0;
	for (int i = 0; i < num_alpha; ++i) {
		if (grp_id.find(grp_vec[i]) != grp_id.end()) {
			coef_prec(i, i) *= lambda;
			gig_chi += (coef[i] - coef_mean[i]) * (coef[i] - coef_mean[i]) * coef_prec(i, i);
			mn_size++;
		}
	}
	lambda = sim_gig(1, shape - mn_size / 2, 2 * rate, gig_chi, rng)[0];
}

// Generating contemporaneous lambda of Minnesota-SV
// 
// @param lambda lambda1 or lambda2
// @param shape Gamma prior shape
// @param rate Gamma prior rate
// @param coef_vec Coefficients vector
// @param coef_mean Prior mean
// @param coef_prec Prior precision matrix
// @param rng boost rng
inline void minnesota_contem_lambda(double& lambda, double& shape, double& rate, Eigen::Ref<Eigen::VectorXd> coef,
														 				Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::MatrixXd& coef_prec,
														 				boost::random::mt19937& rng) {
	coef_prec.diagonal() *= lambda;
	// double gig_chi = ((coef - coef_mean).array().square() / coef_prec.diagonal().array()).sum();
	// double gig_chi = ((coef - coef_mean).array().square()).sum();
	double gig_chi = (coef - coef_mean).squaredNorm();
	lambda = sim_gig(1, shape - coef.size() / 2, 2 * rate, gig_chi, rng)[0];
	coef_prec.diagonal() /= lambda;
}

// Draw d_i in D from cholesky decomposition of precision matrix
// 
// @param diag_vec d_i vector
// @param shape IG shape of d_i prior
// @param scl IG scale of d_i prior
// @param ortho_latent Residual matrix of triangular equation
// @param rng boost rng
inline void reg_ldlt_diag(Eigen::Ref<Eigen::VectorXd> diag_vec, Eigen::VectorXd& shape, Eigen::VectorXd& scl,
													Eigen::MatrixXd& ortho_latent, boost::random::mt19937& rng) {
	int num_design = ortho_latent.rows();
	for (int i = 0; i < diag_vec.size(); ++i) {
		// diag_vec[i] = 1 / gamma_rand(
    //   shape[i] + num_design / 2,
		// 	1 / (scl[i] + ortho_latent.col(i).squaredNorm() / 2),
		// 	rng
    // );
		diag_vec[i] = 1 / gamma_rand(
      shape[i] + num_design / 2,
			1 / (scl[i] + ortho_latent.col(i).squaredNorm() / 2),
			rng
    );
	}
}

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

#endif // BVHARDRAW_H