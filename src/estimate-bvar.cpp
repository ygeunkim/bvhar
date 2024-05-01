#include "minnesota.h"
#include "bvharinterrupt.h"

//' BVAR(p) Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param lag VAR order
//' @param bayes_spec BVAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean) {
	bvhar::BvarSpec mn_spec(bayes_spec);
	std::unique_ptr<bvhar::MinnBvar> mn_obj(new bvhar::MinnBvar(y, lag, mn_spec, include_mean));
	return mn_obj->returnMinnRes();
}

//' BVHAR Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param week VHAR week order
//' @param month VHAR month order
//' @param bayes_spec BVHAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvhar_mn(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean, bool minn_short) {
	std::unique_ptr<bvhar::MinnBvhar> mn_obj;
	if (minn_short) {
		bvhar::BvarSpec bvhar_spec(bayes_spec);
		mn_obj = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(y, week, month, bvhar_spec, include_mean));
	} else {
		bvhar::BvharSpec bvhar_spec(bayes_spec);
		mn_obj = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(y, week, month, bvhar_spec, include_mean));
	}
	return mn_obj->returnMinnRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mh(int num_chains, int num_iter, int num_burn, int thin,
														Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy,
														Rcpp::List param_prior, Rcpp::List param_init,
														Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	std::vector<std::unique_ptr<bvhar::MhMinnesota>> mn_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	Rcpp::List lambda_spec = param_prior["lambda"];
	Rcpp::List psi_spec = param_prior["sigma"];
	bvhar::MhMinnSpec mn_spec(lambda_spec, psi_spec);
  for (int i = 0; i < num_chains; ++i) {
		Rcpp::List init_spec = param_init[i];
		bvhar::MhMinnInits mn_init(init_spec);
		mn_objs[i].reset(new bvhar::MhMinnesota(num_iter, mn_spec, mn_init, x, y, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
		mn_objs[i]->computePosterior();
	}
	auto run_mh = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mn_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			mn_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_mh(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_mh(chain);
		}
	}
	return Rcpp::wrap(res);
}

//' BVAR(p) Point Estimates based on Nonhierarchical Matrix Normal Prior
//' 
//' Point estimates for Ghosh et al. (2018) nonhierarchical model for BVAR.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param U Positive definite matrix, covariance matrix corresponding to the column of the model parameter B
//' 
//' @details
//' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  int num_design = y.rows();
  int dim = y.cols();
  if (U.rows() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  if (U.cols() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  Eigen::MatrixXd prec_mat = (x.transpose() * x + U); // MN precision
  Eigen::MatrixXd mn_scale_mat = prec_mat.inverse(); // MN scale 1 = inverse of precision
  Eigen::MatrixXd coef_mat = mn_scale_mat * x.transpose() * y; // MN mean
  Eigen::MatrixXd yhat = x * coef_mat; // x %*% bhat
  Eigen::MatrixXd Is = Eigen::MatrixXd::Identity(num_design, num_design);
  Eigen::MatrixXd scale_mat = y.transpose() * (Is - x * mn_scale_mat * x.transpose()) * y; // IW scale
  return Rcpp::List::create(
    Rcpp::Named("mnmean") = coef_mat,
    Rcpp::Named("mnprec") = prec_mat,
    Rcpp::Named("fitted") = yhat,
    Rcpp::Named("iwscale") = scale_mat,
    Rcpp::Named("iwshape") = num_design - dim - 1
  );
}
