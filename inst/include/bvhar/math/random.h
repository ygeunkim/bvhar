#ifndef BVHARSIM_H
#define BVHARSIM_H

#include "../core/common.h"
#include <vector>
#include <limits>

#ifdef USE_RCPP
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig, boost::random::mt19937& rng);

Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method);

Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale_v);

Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape);
#endif
namespace bvhar {

// Log quasi-density of GIG
// 
// @param x postivie support
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline double dgig_quasi(double x, double lambda, double beta) {
	return (lambda - 1) * log(x) - beta * (x + 1 / x) / 2;
}

// Compute mode of quasi-density of GIG
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline double dgig_mode(double lambda, double beta) {
	if (lambda < 1) {
		return beta / (sqrt((1 - lambda) * (1 - lambda) + beta * beta) + 1 - lambda);
	}
	return (sqrt((lambda - 1) * (lambda - 1) + beta * beta) - 1 + lambda) / beta;
}

#ifdef USE_RCPP
// Generate MN(M, U, V)
// @param mat_mean Mean matrix M
// @param mat_scale_u First scale matrix U
// @param mat_scale_v Second scale matrix V
// @param prec If true, use mat_scale_u as inverse of U.
inline Eigen::MatrixXd sim_mn(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u, const Eigen::MatrixXd& mat_scale_v,
															bool prec) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  Eigen::MatrixXd chol_scale_v = mat_scale_v.llt().matrixU(); // V = U_vTU_v
  Eigen::MatrixXd mat_norm(num_rows, num_cols); // standard normal
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = norm_rand();
    }
  }
	if (prec) {
		// U^(-1) = LLT => U = LT^(-1) L^(-1)
		return mat_mean + mat_scale_u.llt().matrixU().solve(mat_norm * chol_scale_v); // M + LT^(-1) X U_v ~ MN(M, LT^(-1) L^(-1) = U, U_vT U_v = V)
	}
	Eigen::MatrixXd chol_scale_u = mat_scale_u.llt().matrixL(); // U = LLT
	return mat_mean + chol_scale_u * mat_norm * chol_scale_v; // M + L X U_v ~ MN(M, LLT = U, U_vT U_v = V)
}

// Generate Lower Triangular Matrix of IW
// 
// This function generates \eqn{A = L (Q^{-1})^T}.
// 
// @param mat_scale Scale matrix of IW
// @param shape Shape of IW
// @details
// This function is the internal function for IW sampling and MNIW sampling functions.
inline Eigen::MatrixXd sim_iw_tri(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
	if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim); // upper triangular bartlett decomposition
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i)); // diagonal: qii^2 ~ chi^2(nu - i + 1)
  }
  for (int i = 0; i < dim - 1; i ++) {
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = norm_rand(); // upper triangular (j > i) ~ N(0, 1)
    }
  }
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
  // return chol_scale * mat_bartlett.inverse().transpose(); // lower triangular
	return chol_scale * mat_bartlett.transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(dim, dim)); // lower triangular
}

inline Eigen::MatrixXd sim_inv_wishart(const Eigen::MatrixXd& mat_scale, double shape) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  return res;
}

// Generate MNIW(M, U, Psi, nu)
// 
// @param mat_mean Mean matrix M
// @param mat_scale_u First scale matrix U
// @param mat_scale Inverse wishart scale matrix Psi
// @param shape Inverse wishart shape
// @param prec If true, use mat_scale_u as \eqn{U^{-1}}
inline std::vector<Eigen::MatrixXd> sim_mn_iw(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u,
																			 				const Eigen::MatrixXd& mat_scale, double shape, bool prec) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd mat_scale_v = chol_res * chol_res.transpose();
	std::vector<Eigen::MatrixXd> res(2);
	res[0] = sim_mn(mat_mean, mat_scale_u, mat_scale_v, prec);
	res[1] = mat_scale_v;
	return res;
}

// Generate Lower Triangular Matrix of Wishart
// 
// This function generates \eqn{A = L (Q^{-1})^T}.
// 
// @param mat_scale Scale matrix of Wishart
// @param shape Shape of Wishart
// @details
// This function generates Wishart random matrix.
inline Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i));
  }
  for (int i = 1; i < dim; i++) {
    for (int j = 0; j < i; j++) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(mat_scale);
  Eigen::MatrixXd chol_scale = lltOfscale.matrixL();
  Eigen::MatrixXd chol_res = chol_scale * mat_bartlett;
  return chol_res * chol_res.transpose();
}

// AR-Mehod for non-concave part
// 
// Algorithm 1 of Hormann and Leydold
// 0 <= lambda < 1 and 0 < beta <= 2 / 3 * sqrt(1 - lambda)
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline void rgig_nonconcave(Eigen::VectorXd& res, int num_sim, double lambda, double beta) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double x0, xstar, k1, k2, k3, A1, A2, A3;
	x0 = beta / (1 - lambda); // subdomain (0, x0)
	xstar = std::max(x0, 2 / beta);
	k1 = exp(dgig_quasi(mode, lambda, beta));
	A1 = k1 * x0;
	if (x0 < 2 / beta) { // subdomain (x0, 2 / beta)
		k2 = exp(-beta);
		if (lambda == 0) {
			A2 = k2 * log(2 / (beta * beta));
		} else {
			A2 = k2 * (pow(2 / beta, lambda) - pow(x0, lambda)) / lambda;
		}
	} else {
		k2 = 0;
		A2 = 0;
	}
	k3 = pow(xstar, lambda - 1);  // subdomain (xstar, inf)
	A3 = 2 * k3 * exp(-xstar * beta / 2) / beta;
	double A = A1 + A2 + A3;
	bool rejected;
	double draw_unif, draw_prop, cand, ar_const;
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_prop = unif_rand(0, A);
			if (draw_prop <= A1) { // subdomain (0, x0)
				cand = x0 * draw_prop / A1;
				ar_const = log(k1);
			} else if (draw_prop <= A1 + A2) { // subdomain (x0, 2 / beta)
				draw_prop -= A1;
				if (lambda == 0) {
					cand = beta * exp(draw_prop * exp(beta));
				} else {
					cand = pow(pow(x0, lambda) + draw_prop * lambda / k2, 1 / lambda);
				}
				ar_const = log(k2) + (lambda - 1) * log(cand);
			} else { // subdomain (xstar, inf)
				draw_prop -= (A1 + A2);
				cand = -2 * log(exp(-xstar * beta / 2) - draw_prop * beta / (2 * k3)) / beta;
				ar_const = log(k3) - cand * beta / 2;
			}
			draw_unif = unif_rand(0, 1);
			rejected = log(draw_unif) + ar_const > dgig_quasi(cand, lambda, beta);
		}
		res[i] = cand;
	}
}

// Ratio-of-Uniforms without Mode Shift
// 
// Algorithm 2 of Hormann and Leydold
// 0 <= lambda <= 1 and min(1 / 2, 2 / 3 * sqrt(1 - lambda)) <= beta <= 1
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline void rgig_without_mode(Eigen::VectorXd& res, int num_sim, double lambda, double beta) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double mode_x = (1 + lambda + sqrt((1 + lambda) * (1 + lambda) + beta * beta)) / beta; // argmax of x g(x)
	double bound_y = dgig_quasi(mode, lambda, beta) / 2; // To normalize g
	double bound_x = exp(log(mode_x) + dgig_quasi(mode_x, lambda, beta) / 2 - bound_y);
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(0, bound_x);
			draw_y = unif_rand(0, 1);
			cand = draw_x / draw_y;
			rejected = log(draw_y) > dgig_quasi(cand, lambda, beta) / 2 - bound_y; // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
}

// Ratio-of-Uniforms with Mode Shift
// 
// Algorithm 2 of Hormann and Leydold
// lambda > 1 and beta > 1
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline void rgig_with_mode(Eigen::VectorXd& res, int num_sim, double lambda, double beta) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double quad_coef = -2 * (lambda + 1) / beta - mode;
	double lin_coef = 2 * mode * (lambda - 1) / beta - 1;
	double p = lin_coef - quad_coef * quad_coef / 3;
	double q = 2 * quad_coef * quad_coef * quad_coef / 27 - quad_coef * lin_coef / 3 + mode;
	double phi = acos(-q * sqrt(-27 / (p * p * p)) / 2);
	double arg_x_neg = sqrt(-p * 4 / 3) * cos(phi / 3 + M_PI * 4 / 3) - quad_coef / 3;
	double arg_x_pos = sqrt(-p * 4 / 3) * cos(phi / 3) - quad_coef / 3;
	double bound_y = dgig_quasi(mode, lambda, beta) / 2; // use as normalize factor
	double bound_x_neg = (arg_x_neg - mode) * exp(dgig_quasi(arg_x_neg, lambda, beta) / 2 - bound_y);
	double bound_x_pos = (arg_x_pos - mode) * exp(dgig_quasi(arg_x_pos, lambda, beta) / 2 - bound_y);
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(bound_x_neg, bound_x_pos);
			draw_y = unif_rand(0, 1); // U(0, 1) since g has been normalized
			cand = draw_x / draw_y + mode;
			if (cand > 0) {
				rejected = log(draw_y) > dgig_quasi(cand, lambda, beta) / 2 - bound_y; // Check if U <= g(y) / unif(y)
			} else {
				rejected = true; // cand can be negative
			}
		}
		res[i] = cand;
	}
}

// Generate Generalized Inverse Gaussian Distribution
// 
// This function samples GIG(lambda, psi, chi) random variates.
// Also, we follow boundaries chosen by GIGrvg package.
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param psi Second parameter of GIG
// @param chi Third parameter of GIG
inline Eigen::VectorXd sim_gig(int num_sim, double lambda, double psi, double chi) {
	// if (psi <= 0 || chi <= 0) {
	// 	STOP("Wrong 'psi' and 'chi' range.");
	// }
	Eigen::VectorXd res(num_sim);
	double abs_lam = abs(lambda); // If lambda < 0, use 1 / X as the result
	// double alpha = sqrt(psi / chi); // rate parameter of quasi-density: scale the result
	double beta = sqrt(psi * chi); // second parameter of quasi-density
	if (beta < 8 * std::numeric_limits<double>::epsilon()) {
		// Handle round-off error following GIGrvg
		if (lambda > 0) {
			for (int i = 0; i < num_sim; ++i) {
				res[i] = gamma_rand(abs_lam, 2 / psi); // GIG(shape, 2 * rate, 0) <=> Gamma(shape, rate)
			}
			return res;
		} else if (lambda < 0) {
			for (int i = 0; i < num_sim; ++i) {
				res[i] = 1 / gamma_rand(abs_lam, 2 / chi); // GIG(-shape, 0, 2 * scale) <=> Inverse-Gamma(shape, scale)
			}
			return res;
		}
	}
	// if (chi < 15 * std::numeric_limits<double>::epsilon()) {
	// 	for (int i = 0; i < num_sim; ++i) {
	// 		res[i] = gamma_rand(abs_lam, 2 / psi); // GIG(shape, 2 * rate, 0) <=> Gamma(shape, rate)
	// 	}
	// 	if (lambda < 0) {
	// 		return res.cwiseInverse();
	// 	}
	// 	return res;
	// }
	// if (psi < 15 * std::numeric_limits<double>::epsilon()) {
	// 	for (int i = 0; i < num_sim; ++i) {
	// 		res[i] = 1 / gamma_rand(abs_lam, 2 / chi); // GIG(-shape, 0, 2 * scale) <=> Inverse-Gamma(shape, scale)
	// 	}
	// 	if (lambda < 0) {
	// 		return res.cwiseInverse();
	// 	}
	// 	return res;
	// }
	// double alpha = sqrt(psi / chi); // rate parameter of quasi-density: scale the result
	// double beta = sqrt(psi * chi); // second parameter of quasi-density
	if (abs_lam > 2 || beta > 3) {
		rgig_with_mode(res, num_sim, abs_lam, beta); // with mode shift
	} else if (abs_lam >= 1 - 9 * beta * beta / 4 || beta > .2) {
		rgig_without_mode(res, num_sim, abs_lam, beta); // without mode shift
	} else if (beta > 0) {
		rgig_nonconcave(res, num_sim, abs_lam, beta); // non-T_(-1/2)-concave part
	} else {
		STOP("Wrong parameter ranges for quasi GIG density: lambda = %g, psi = %g, chi = %g", lambda, psi, chi);
	}
	if (lambda < 0) {
		// res = res.cwiseInverse();
		return sqrt(chi / psi) * res.cwiseInverse();
	}
	return res * sqrt(chi / psi);
}
#endif

// Generate MN(M, U, V)
inline Eigen::MatrixXd sim_mn(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u, const Eigen::MatrixXd& mat_scale_v,
															bool prec, boost::random::mt19937& rng) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  Eigen::MatrixXd chol_scale_v = mat_scale_v.llt().matrixU(); // V = U_vTU_v
  Eigen::MatrixXd mat_norm(num_rows, num_cols); // standard normal
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = normal_rand(rng);
    }
  }
	if (prec) {
		return mat_mean + mat_scale_u.llt().matrixU().solve(mat_norm * chol_scale_v); // M + U_u^(-1) X U_v ~ MN(M, U_u^(-1) U_u^(-1)T = U, U_vT U_v = V)
	}
	Eigen::MatrixXd chol_scale_u = mat_scale_u.llt().matrixL(); // U = LLT
	return mat_mean + chol_scale_u * mat_norm * chol_scale_v; // M + L X U_v ~ MN(M, LLT = U, U_vT U_v = V)
}

// Generate Lower Triangular Matrix of IW
inline Eigen::MatrixXd sim_iw_tri(const Eigen::MatrixXd& mat_scale, double shape, boost::random::mt19937& rng) {
  int dim = mat_scale.cols();
	if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim); // upper triangular bartlett decomposition
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i, rng)); // diagonal: qii^2 ~ chi^2(nu - i + 1)
  }
  for (int i = 0; i < dim - 1; i ++) {
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = normal_rand(rng); // upper triangular (j > i) ~ N(0, 1)
    }
  }
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
	return chol_scale * mat_bartlett.transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(dim, dim)); // lower triangular
}

// Generate MNIW(M, U, Psi, nu)
inline std::vector<Eigen::MatrixXd> sim_mn_iw(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u,
																			 				const Eigen::MatrixXd& mat_scale, double shape, bool prec, boost::random::mt19937& rng) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape, rng);
  Eigen::MatrixXd mat_scale_v = chol_res * chol_res.transpose();
	std::vector<Eigen::MatrixXd> res(2);
	res[0] = sim_mn(mat_mean, mat_scale_u, mat_scale_v, prec, rng);
	res[1] = mat_scale_v;
	return res;
}

// AR-Mehod for non-concave part
inline void rgig_nonconcave(Eigen::VectorXd& res, int num_sim, double lambda, double beta, boost::random::mt19937& rng) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double x0, xstar, k1, k2, k3, A1, A2, A3;
	x0 = beta / (1 - lambda); // subdomain (0, x0)
	xstar = std::max(x0, 2 / beta);
	k1 = exp(dgig_quasi(mode, lambda, beta));
	A1 = k1 * x0;
	if (x0 < 2 / beta) { // subdomain (x0, 2 / beta)
		k2 = exp(-beta);
		if (lambda == 0) {
			A2 = k2 * log(2 / (beta * beta));
		} else {
			A2 = k2 * (pow(2 / beta, lambda) - pow(x0, lambda)) / lambda;
		}
	} else {
		k2 = 0;
		A2 = 0;
	}
	k3 = pow(xstar, lambda - 1);  // subdomain (xstar, inf)
	A3 = 2 * k3 * exp(-xstar * beta / 2) / beta;
	double A = A1 + A2 + A3;
	bool rejected;
	double draw_unif, draw_prop, cand, ar_const;
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		int iter_while = 0;
		while (rejected && iter_while <= std::numeric_limits<int>::max() / 2) {
			draw_prop = A * unif_rand(rng);
			if (draw_prop <= A1) { // subdomain (0, x0)
				cand = x0 * draw_prop / A1;
				ar_const = log(k1);
			} else if (draw_prop <= A1 + A2) { // subdomain (x0, 2 / beta)
				draw_prop -= A1;
				if (lambda == 0) {
					cand = beta * exp(draw_prop * exp(beta));
				} else {
					cand = pow(pow(x0, lambda) + draw_prop * lambda / k2, 1 / lambda);
				}
				ar_const = log(k2) + (lambda - 1) * log(cand);
			} else { // subdomain (xstar, inf)
				draw_prop -= (A1 + A2);
				cand = -2 * log(exp(-xstar * beta / 2) - draw_prop * beta / (2 * k3)) / beta;
				ar_const = log(k3) - cand * beta / 2;
			}
			draw_unif = unif_rand(rng);
			rejected = log(draw_unif) + ar_const > dgig_quasi(cand, lambda, beta);
			++iter_while;
		}
		res[i] = cand;
	}
}

// Ratio-of-Uniforms without Mode Shift
inline void rgig_without_mode(Eigen::VectorXd& res, int num_sim, double lambda, double beta, boost::random::mt19937& rng) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double mode_x = (1 + lambda + sqrt((1 + lambda) * (1 + lambda) + beta * beta)) / beta; // argmax of x g(x)
	double bound_y = dgig_quasi(mode, lambda, beta) / 2; // To normalize g
	double bound_x = exp(log(mode_x) + dgig_quasi(mode_x, lambda, beta) / 2 - bound_y);
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		int iter_while = 0;
		while (rejected && iter_while <= std::numeric_limits<int>::max() / 2) {
			draw_x = bound_x * unif_rand(rng);
			draw_y = unif_rand(rng);
			cand = draw_x / draw_y;
			++iter_while;
			rejected = log(draw_y) > dgig_quasi(cand, lambda, beta) / 2 - bound_y; // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
}


// Ratio-of-Uniforms with Mode Shift
inline void rgig_with_mode(Eigen::VectorXd& res, int num_sim, double lambda, double beta, boost::random::mt19937& rng) {
	double mode = dgig_mode(lambda, beta); // argmax of g(x)
	double quad_coef = -2 * (lambda + 1) / beta - mode;
	double lin_coef = 2 * mode * (lambda - 1) / beta - 1;
	double p = lin_coef - quad_coef * quad_coef / 3;
	double q = 2 * quad_coef * quad_coef * quad_coef / 27 - quad_coef * lin_coef / 3 + mode;
	double phi = acos(-q * sqrt(-27 / (p * p * p)) / 2);
	double arg_x_neg = sqrt(-p * 4 / 3) * cos(phi / 3 + M_PI * 4 / 3) - quad_coef / 3;
	double arg_x_pos = sqrt(-p * 4 / 3) * cos(phi / 3) - quad_coef / 3;
	double bound_y = dgig_quasi(mode, lambda, beta) / 2; // use as normalize factor
	double bound_x_neg = (arg_x_neg - mode) * exp(dgig_quasi(arg_x_neg, lambda, beta) / 2 - bound_y);
	double bound_x_pos = (arg_x_pos - mode) * exp(dgig_quasi(arg_x_pos, lambda, beta) / 2 - bound_y);
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		int iter_while = 0;
		while (rejected && iter_while <= std::numeric_limits<int>::max() / 2) {
			draw_x = (bound_x_pos - bound_x_neg) * unif_rand(rng) + bound_x_neg;
			draw_y = unif_rand(rng); // U(0, 1) since g has been normalized
			cand = draw_x / draw_y + mode;
			if (cand > 0) {
				rejected = log(draw_y) > dgig_quasi(cand, lambda, beta) / 2 - bound_y; // Check if U <= g(y) / unif(y)
			} else {
				rejected = true; // cand can be negative
			}
			++iter_while;
		}
		res[i] = cand;
	}
}

// Generate Generalized Inverse Gaussian Distribution
inline Eigen::VectorXd sim_gig(int num_sim, double lambda, double psi, double chi, boost::random::mt19937& rng) {
	Eigen::VectorXd res(num_sim);
	double abs_lam = abs(lambda); // GIG(lambda, psi, chi) == 1 / GIG(-lambda, chi, psi)
	// double alpha = sqrt(psi / chi); // rate parameter of quasi-density: scale the result by X / alpha
	if (chi == 0 || std::isnan(chi)) {
		chi = std::numeric_limits<double>::min();
	}
	double beta = sqrt(psi * chi); // second parameter of quasi-density
	if (beta < 8 * std::numeric_limits<double>::epsilon()) {
		// Handle round-off error following GIGrvg
		if (lambda > 0) {
			for (int i = 0; i < num_sim; ++i) {
				res[i] = gamma_rand(abs_lam, 2 / psi, rng); // GIG(lambda, psi, 0) == Gamma(lambda, psi / 2)
			}
			return res;
		} else if (lambda < 0) {
			for (int i = 0; i < num_sim; ++i) {
				res[i] = 1 / gamma_rand(abs_lam, 2 / chi, rng); // 1 / GIG(abs(lambda), chi, 0) == 1 / Gamma(abs(lambda), chi / 2)
			}
			return res;
		}
	}
	// double alpha = sqrt(psi / chi); // rate parameter of quasi-density: scale the result by X / alpha
	// double beta = sqrt(psi * chi); // second parameter of quasi-density
	if (abs_lam > 2 || beta > 3) {
		rgig_with_mode(res, num_sim, abs_lam, beta, rng); // with mode shift
	} else if (abs_lam >= 1 - 9 * beta * beta / 4 || beta > .2) {
		rgig_without_mode(res, num_sim, abs_lam, beta, rng); // without mode shift
	} else if (beta > 0) {
		rgig_nonconcave(res, num_sim, abs_lam, beta, rng); // non-T_(-1/2)-concave part
	} else {
		STOP("Wrong parameter ranges for quasi GIG density: lambda = %g, psi = %g, chi = %g", lambda, psi, chi);
	}
	if (lambda < 0) {
		// res = res.cwiseInverse();
		return sqrt(chi / psi) * res.cwiseInverse();
	}
	return res * sqrt(chi / psi);
}

// Generate Inverse Gaussian Distribution
// This function generates one Inverse Gaussian random number with mu (mean) and lambda (shape).
inline double sim_invgauss(double mean, double shape, boost::random::mt19937& rng) {
	double y = mean * chisq_rand(1, rng) / (2 * shape);
	// double cand = mean + mean * y - mean * sqrt(2 * y + y * y);
	double cand_fac = 1 + y - sqrt(2 * y + y * y);
	// double cand = mean * (1 + y - sqrt(2 * y + y * y));
	// if (unif_rand(0, 1, rng) <= mean / (mean + cand)) {
	if (unif_rand(rng) <= 1 / (1 + cand_fac)) {
		// return cand;
		return mean * cand_fac;
	}
	// return mean * mean / cand;
	return mean / cand_fac;
}

} //namespace bvhar

#endif // BVHARSIM_H
