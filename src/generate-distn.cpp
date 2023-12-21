#include <RcppEigen.h>
#include "bvharprob.h"

//' Generate Multivariate Normal Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate Normal Random Vector using Cholesky Decomposition
//' 
//' This function samples n x muti-dimensional normal random matrix with using Cholesky decomposition.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @details
//' This function computes \eqn{\Sigma^{1/2}} by choleksy decomposition.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  // Eigen::LLT<Eigen::MatrixXd> lltOfscale(sig);
  // Eigen::MatrixXd sig_sqrt = lltOfscale.matrixU(); // use upper because now dealing with row vectors
  res = standard_normal * sig.llt().matrixU(); // use upper because now dealing with row vectors
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate t Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param df Degrees of freedom
//' @param mu Location vector
//' @param sig Scale matrix
//' @param method Method to compute \eqn{\Sigma^{1/2}}. 1: spectral decomposition, 2: Cholesky.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd res(num_sim, dim);
  switch (method) {
  case 1:
    res = sim_mgaussian(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  case 2:
    res = sim_mgaussian_chol(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  default:
    Rcpp::stop("Invalid 'method' option.");
  }
  for (int i = 0; i < num_sim; i++) {
    res.row(i) *= sqrt(df / chisq_rand(df));
  }
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Matrix Normal Random Matrix
//' 
//' This function samples one matrix gaussian matrix.
//' 
//' @param mat_mean Mean matrix
//' @param mat_scale_u First scale matrix
//' @param mat_scale_v Second scale matrix
//' @details
//' Consider n x k matrix \eqn{Y_1, \ldots, Y_n \sim MN(M, U, V)} where M is n x k, U is n x n, and V is k x k.
//' 
//' 1. Lower triangular Cholesky decomposition: \eqn{U = P P^T} and \eqn{V = L L^T}
//' 2. Standard normal generation: s x m matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. \eqn{Y_i = M + P Z_i L^T}
//' 
//' This function only generates one matrix, i.e. \eqn{Y_1}.
//' @return One n x k matrix following MN distribution.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, 
                                Eigen::MatrixXd mat_scale_u, 
                                Eigen::MatrixXd mat_scale_v) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  if (mat_scale_u.rows() != mat_scale_u.cols()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (num_rows != mat_scale_u.rows()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_scale_v.rows() != mat_scale_v.cols()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  if (num_cols != mat_scale_v.rows()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscaleu(mat_scale_u);
  Eigen::LLT<Eigen::MatrixXd> lltOfscalev(mat_scale_v);
  // Cholesky decomposition (lower triangular)
  Eigen::MatrixXd chol_scale_u = lltOfscaleu.matrixL();
  Eigen::MatrixXd chol_scale_v = lltOfscalev.matrixL();
  // standard normal
  Eigen::MatrixXd mat_norm(num_rows, num_cols);
  // Eigen::MatrixXd res(num_rows, num_cols, num_sim);
  Eigen::MatrixXd res(num_rows, num_cols);
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = norm_rand();
    }
  }
  res = mat_mean + chol_scale_u * mat_norm * chol_scale_v.transpose();
  return res;
}

//' Generate Lower Triangular Matrix of IW
//' 
//' This function generates \eqn{A = L (Q^{-1})^T}.
//' 
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @details
//' This function is the internal function for IW sampling and MNIW sampling functions.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw_tri(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    Rcpp::stop("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  // upper triangular bartlett decomposition
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    // diagonal
    mat_bartlett(i, i) = sqrt(chisq_rand(shape - (double)i)); // qii^2 ~ chi^2(nu - i + 1)
  }
  // upper triangular (j > i) ~ N(0, 1)
  for (int i = 0; i < dim - 1; i ++) {
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
  // cholesky decomposition (lower triangular)
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(mat_scale);
  Eigen::MatrixXd chol_scale = lltOfscale.matrixL();
  // lower triangular
  Eigen::MatrixXd chol_res = chol_scale * mat_bartlett.inverse().transpose();
  return chol_res;
}

//' Generate Inverse-Wishart Random Matrix
//' 
//' This function samples one matrix IW matrix.
//' 
//' @param mat_scale Scale matrix
//' @param shape Shape
//' @details
//' Consider \eqn{\Sigma \sim IW(\Psi, \nu)}.
//' 
//' 1. Upper triangular Bartlett decomposition: k x k matrix \eqn{Q = [q_{ij}]} upper triangular with
//'     1. \eqn{q_{ii}^2 \chi_{\nu - i + 1}^2}
//'     2. \eqn{q_{ij} \sim N(0, 1)} with i < j (upper triangular)
//' 2. Lower triangular Cholesky decomposition: \eqn{\Psi = L L^T}
//' 3. \eqn{A = L (Q^{-1})^T}
//' 4. \eqn{\Sigma = A A^T \sim IW(\Psi, \nu)}
//' @return One k x k matrix following IW distribution
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  return res;
}

//' Generate Normal-IW Random Family
//' 
//' This function samples normal inverse-wishart matrices.
//' 
//' @param num_sim Number to generate
//' @param mat_mean Mean matrix of MN
//' @param mat_scale_u First scale matrix of MN
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @details
//' Consider \eqn{(Y_i, \Sigma_i) \sim MIW(M, U, \Psi, \nu)}.
//' 
//' 1. Generate upper triangular factor of \eqn{\Sigma_i = C_i C_i^T} in the upper triangular Bartlett decomposition.
//' 2. Standard normal generation: n x k matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. Lower triangular Cholesky decomposition: \eqn{U = P P^T}
//' 4. \eqn{A_i = M + P Z_i C_i^T}
//' @return List of MN and IW matrices.
//' Multiple samples are column-stacked.
//' @export
// [[Rcpp::export]]
Rcpp::List sim_mniw(int num_sim,
                    Eigen::MatrixXd mat_mean, 
                    Eigen::MatrixXd mat_scale_u, 
                    Eigen::MatrixXd mat_scale, 
                    double shape) {
  int ncol_mn = mat_mean.cols();
  int nrow_mn = mat_mean.rows();
  int dim_iw = mat_scale.cols();
  if (dim_iw != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd chol_res(dim_iw, dim_iw);
  Eigen::MatrixXd mat_scale_v(dim_iw, dim_iw);
  // result matrices: bind in column wise
  Eigen::MatrixXd res_mn(nrow_mn, num_sim * ncol_mn); // [Y1, Y2, ..., Yn]
  Eigen::MatrixXd res_iw(dim_iw, num_sim * dim_iw); // [Sigma1, Sigma2, ... Sigma2]
  for (int i = 0; i < num_sim; i++) {
    chol_res = sim_iw_tri(mat_scale, shape);
    mat_scale_v = chol_res * chol_res.transpose();
    res_iw.block(0, i * dim_iw, dim_iw, dim_iw) = mat_scale_v;
    // MN(mat_mean, mat_scale_u, mat_scale_v)
    res_mn.block(0, i * ncol_mn, nrow_mn, ncol_mn) = sim_matgaussian(
      mat_mean, 
      mat_scale_u, 
      mat_scale_v
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("mn") = res_mn,
    Rcpp::Named("iw") = res_iw
  );
}

//' Generate Lower Triangular Matrix of Wishart
//' 
//' This function generates \eqn{A = L (Q^{-1})^T}.
//' 
//' @param mat_scale Scale matrix of Wishart
//' @param shape Shape of Wishart
//' @details
//' This function generates Wishart random matrix.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    Rcpp::stop("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(chisq_rand(shape - (double)i));
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

//' Quasi-density of GIG
//' 
//' @param x postivie support
//' @param lambda Index of modified Bessel function of third kind.
//' @param beta Square of the multiplication of the other two parameters.
//' 
//' @noRd
// [[Rcpp::export]]
double dgig_quasi(double x, double lambda, double beta) {
	return pow(x, lambda - 1) * exp(-beta * (x + 1 / x) / 2);
}

//' AR-Mehod for non-concave part
//' 
//' @param num_sim Number to generate process
//' @param lambda Index of modified Bessel function of third kind.
//' @param beta Square of the multiplication of the other two parameters.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd rgig_nonconcave(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double mode = beta / (sqrt((1 - lambda) * (1 - lambda) + beta * beta) + 1 - lambda); // argmax of g(x)
	double x0, xstar, k1, k2, k3, A1, A2, A3;
	x0 = beta / (1 - beta); // subdomain (0, x0)
	xstar = std::max(x0, 2 / beta);
	k1 = dgig_quasi(mode, lambda, beta);
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
	// double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_unif = unif_rand(0, 1);
			draw_prop = unif_rand(0, A);
			if (draw_prop <= A1) {
				cand = x0 * draw_prop / A1;
				ar_const = k1;
			} else if (draw_prop <= A1 + A2) {
				draw_prop -= A1;
				cand = pow(pow(x0, lambda) + draw_prop * lambda / k2, 1 / lambda);
				ar_const = k2 * pow(cand, lambda - 1);
			} else {
				draw_prop -= A1 + A2;
				cand = -2 * log(exp(-xstar * beta / 2) - draw_prop * beta / (2 * k3));
				ar_const = k3 * exp(-cand * beta / 2);
			}
			rejected = draw_unif * ar_const > dgig_quasi(cand, lambda, beta);
		}
		res[i] = cand;
	}
	return res;
}

//' Ratio-of-Uniforms without Mode Shift
//' 
//' @param num_sim Number to generate process
//' @param lambda Index of modified Bessel function of third kind.
//' @param beta Square of the multiplication of the other two parameters.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd rgig_without_mode(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double arg_y = beta / (sqrt((1 - lambda) * (1 - lambda) + beta * beta) + 1 - lambda); // argmax of g(x)
	double arg_x = (1 + lambda + sqrt((1 + lambda) * (1 + lambda) + beta * beta)) / beta; // argmax of x g(x)
	double bound_y = sqrt(dgig_quasi(arg_y, lambda, beta)); // max sqrt(g(x))
	double bound_x = arg_x * sqrt(dgig_quasi(arg_x, lambda, beta)); // max x*sqrt(g(x))
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(0, bound_x);
			draw_y = unif_rand(0, bound_y);
			cand = draw_x / draw_y;
			rejected = draw_y * draw_y > dgig_quasi(cand, lambda, beta); // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
	return res;
}

//' Ratio-of-Uniforms with Mode Shift
//' 
//' @param num_sim Number to generate process
//' @param lambda Index of modified Bessel function of third kind.
//' @param beta Square of the multiplication of the other two parameters.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd rgig_with_mode(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double arg_y = (sqrt((1 - lambda) * (1 - lambda) + beta * beta) - 1 + lambda) / beta; // argmax of g(x)
	double quad_coef = -2 * (lambda + 1) / beta - arg_y;
	double lin_coef = 2 * arg_y * (lambda - 1) / beta - 1;
	double p = lin_coef - quad_coef * quad_coef / 3;
	double q = 2 * quad_coef * quad_coef * quad_coef / 27 - quad_coef * lin_coef * arg_y / 3 + arg_y;
	double phi = acos(-q * sqrt(-27 / (p * p * p)) / 2);
	double arg_x_neg = sqrt(-p * 4 / 3) * cos(phi / 3 + M_PI * 4 / 3) - quad_coef / 3;
	double arg_x_pos = sqrt(-p * 4 / 3) * cos(phi / 3) - quad_coef / 3;
	double bound_y = sqrt(dgig_quasi(arg_y, lambda, beta));
	double bound_x_neg = (arg_x_neg - arg_y) * sqrt(dgig_quasi(arg_x_neg, lambda, beta));
	double bound_x_pos = (arg_x_pos - arg_y) * sqrt(dgig_quasi(arg_x_pos, lambda, beta));
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(bound_x_neg, bound_x_pos);
			draw_y = unif_rand(0, bound_y);
			cand = draw_x / draw_y + arg_y;
			rejected = draw_y * draw_y > dgig_quasi(cand, lambda, beta); // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
	return res;
}

//' Generate Generalized Inverse Gaussian Distribution
//' 
//' This function samples GIG(lambda, psi, chi) random variates.
//' 
//' @param num_sim Number to generate process
//' @param lambda Index of modified Bessel function of third kind.
//' @param psi Second parameter of GIG
//' @param chi Third parameter of GIG
//' 
//' @references Hörmann, W., Leydold, J. Generating generalized inverse Gaussian random variates. Stat Comput 24, 547–557 (2014).
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd sim_gig(int num_sim, double lambda, double psi, double chi) {
	if (psi <= 0 || chi <= 0) {
		Rcpp::stop("Wrong 'psi' and 'chi' range.");
	}
	Eigen::VectorXd res(num_sim);
	double abs_lam = abs(lambda); // If lambda < 0, use 1 / X as the result
	double alpha = sqrt(psi / chi); // scaling parameter of quasi-density: scale the result
	double beta = sqrt(psi * chi); // second parameter of quasi-density
	double quasi_bound = sqrt(1 - lambda) * 2 / 3;
	if (abs_lam < 1 && beta <= quasi_bound) {
		res = rgig_nonconcave(num_sim, lambda, beta) * alpha; // non-T_(-1/2)-concave part
	} else if (abs_lam <= 1 && beta >= std::min(.5, quasi_bound) && beta <= 1) {
		res = rgig_without_mode(num_sim, lambda, beta) * alpha; // without mode shift
	} else if (abs_lam > 1 && beta > 1) {
		res = rgig_with_mode(num_sim, lambda, beta) * alpha; // with mode shift
	} else {
		Rcpp::stop("Wrong parameter ranges for quasi GIG density.");
	}
	if (lambda < 0) {
		res = res.cwiseInverse();
	}
	return res;
}
