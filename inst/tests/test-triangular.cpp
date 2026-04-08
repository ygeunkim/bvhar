#include <catch2/catch_test_macros.hpp>
#include <bvhar/triangular>

BVHAR_LIST get_spec(int prior_type, int dim, int lag, int num_grp) {
	switch (prior_type) {
		case 1: {
			BVHAR_LIST param_prior = BVHAR_CREATE_LIST(
				BVHAR_NAMED("hierarchical") = false,
				BVHAR_NAMED("lambda") = .1,
				BVHAR_NAMED("eps") = 1e-04,
				BVHAR_NAMED("sigma") = Eigen::VectorXd::Ones(dim),
				BVHAR_NAMED("delta") = Eigen::VectorXd::Zero(dim)
			);
			if (lag > 0) {
				param_prior["p"] = lag;
			}
			return param_prior;
		}
		case 2: {
			return BVHAR_CREATE_LIST(
				BVHAR_NAMED("s1") = Eigen::VectorXd::Ones(num_grp),
				BVHAR_NAMED("s2") = Eigen::VectorXd::Ones(num_grp),
				BVHAR_NAMED("slab_shape") = .01,
				BVHAR_NAMED("slab_scl") = .01,
				BVHAR_NAMED("grid_size") = 100
			);
		}
		case 3: {
			return {};
		}
		case 4: {
			BVHAR_LIST param_prior = BVHAR_CREATE_LIST(
				BVHAR_NAMED("hierarchical") = true,
				BVHAR_NAMED("eps") = 1e-04,
				BVHAR_NAMED("sigma") = Eigen::VectorXd::Ones(dim),
				BVHAR_NAMED("delta") = Eigen::VectorXd::Zero(dim),
				BVHAR_NAMED("shape") = .1,
				BVHAR_NAMED("rate") = .1,
				BVHAR_NAMED("grid_size") = 100
			);
			if (lag > 0) {
				param_prior["p"] = lag;
			}
			return param_prior;
		}
		case 5: {
			return BVHAR_CREATE_LIST(
				BVHAR_NAMED("shape_sd") = .01,
				BVHAR_NAMED("group_shape") = .01,
				BVHAR_NAMED("group_scale") = .01,
				BVHAR_NAMED("global_shape") = .01,
				BVHAR_NAMED("global_scale") = .01
			);
		}
		case 6: {
			return BVHAR_CREATE_LIST(
				BVHAR_NAMED("shape") = .01,
				BVHAR_NAMED("scale") = .01,
				BVHAR_NAMED("grid_size") = 100
			);
		}
		case 7: {
			return BVHAR_CREATE_LIST(
				BVHAR_NAMED("grid_shape") = 100,
				BVHAR_NAMED("grid_rate") = 100
			);
		}
	}
	return BVHAR_CREATE_LIST(
		BVHAR_NAMED("grid_shape") = 100,
		BVHAR_NAMED("grid_rate") = 100
	);
}

BVHAR_LIST_OF_LIST run_bvar_cta(
	int num_chains, int nthreads, int num_iter, int num_burn, int thin,
	int dim, int num_data, int lag,
	int group_type,
	int cov_type, int prior_type, int contem_prior_type,
	bool include_mean, bool ggl
) {
	int dim_design = include_mean ? dim * lag + 1 : dim * lag;
	int num_coef = dim * dim_design;
	int num_alpha = include_mean ? num_coef - dim : num_coef;
	int num_eta = dim * (dim - 1) / 2;
	BVHAR_BHRNG dgp_rng(1);
	Eigen::MatrixXd time_series(num_data, dim);
	for (int i = 0; i < num_data; ++i) {
		for (int j = 0; j < dim; ++j) {
			time_series(i, j) = baecon::bvhar::normal_rand(dgp_rng);
		}
	}
	Eigen::MatrixXd x = baecon::bvhar::build_x0(time_series, lag, include_mean);
	Eigen::MatrixXd y = baecon::bvhar::build_y0(time_series, lag, lag + 1);
	Eigen::MatrixXi grp_mat = baecon::bvhar::build_grpmat(lag, dim, group_type);
	Eigen::VectorXi grp_id = grp_mat.unique();
	int num_grp = grp_id.size();
	Eigen::VectorXi own_id = baecon::bvhar::build_own_id(lag, group_type);
	Eigen::VectorXi cross_id = baecon::bvhar::build_cross_id(lag, group_type);
	BVHAR_LIST param_prior = get_spec(prior_type, dim, lag, num_grp);
	BVHAR_LIST contem_prior = get_spec(contem_prior_type, num_eta, 0, 2);
	BVHAR_LIST param_reg = BVHAR_CREATE_LIST(
		BVHAR_NAMED("shape") = Eigen::VectorXd::Constant(dim, 3),
		BVHAR_NAMED("scale") = Eigen::VectorXd::Constant(dim, .01)
	);
	if (cov_type == 2) {
		param_reg["initial_mean"] = BVHAR_CAST_VECTOR(Eigen::VectorXd::Ones(dim));
		param_reg["initial_prec"] = BVHAR_CAST_VECTOR(Eigen::VectorXd::Constant(dim, .1));
	}
	BVHAR_LIST param_intercept = BVHAR_CREATE_LIST(
		BVHAR_NAMED("mean_non") = Eigen::VectorXd::Zero(dim),
		BVHAR_NAMED("sd_non") = .1
	);
	Eigen::VectorXi seed_chain = Eigen::VectorXi::Random(num_chains);
	auto mcmc_run = [&]() -> std::unique_ptr<baecon::bvhar::McmcRun> {
		if (BVHAR_CONTAINS(param_reg, "initial_mean")) {
			if (ggl) {
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, prior_type,
					contem_prior, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain, false, nthreads
				);
			} else {
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, prior_type,
					contem_prior, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain, false, nthreads
				); 
			}
		}
		if (ggl) {
			return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, prior_type,
				contem_prior, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain, false, nthreads
			);
		}
		return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(
			num_chains, num_iter, num_burn, thin, x, y,
			param_reg, param_prior, param_intercept, prior_type,
			contem_prior, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain, false, nthreads
		);
	}();
	return mcmc_run->returnRecords();
}

TEST_CASE("BVAR: Corrected Triangular Algorithm", "[triangular]") {
	int num_chains = 1;
	int nthreads = 1;
	int num_iter = 5;
	int num_burn = 1;
	int thin = 2;
	int dim = 3;
	int num_data = 30;
	int lag = 2;
	bool include_mean = true;
	for (bool ggl : {true, false}) {
		for (int group_type = 1; group_type <= 3; ++group_type) {
			for (int cov_type = 1; cov_type <= 2; ++cov_type) {
				for (int prior_type = 1; prior_type <= 7; ++prior_type) {
					for (int contem_prior_type = 1; contem_prior_type <= 7; ++contem_prior_type) {
						DYNAMIC_SECTION(
							"ggl=" << ggl
								<< ", group_type=" << group_type
								<< ", cov_type=" << cov_type
								<< ", prior_type=" << prior_type
  		        	<< ", contem_prior_type=" << contem_prior_type
						) {
							BVHAR_LIST_OF_LIST res = run_bvar_cta(
								num_chains, nthreads, num_iter, num_burn, thin,
								dim, num_data, lag,
								group_type, cov_type, prior_type, contem_prior_type,
								include_mean, ggl
							);
							REQUIRE(res.size() == num_chains);
							for (int i = 0; i < num_chains; ++i) {
								REQUIRE(BVHAR_CONTAINS(res[i], "alpha_record"));
								REQUIRE(BVHAR_CONTAINS(res[i], "a_record"));
							}
						}
					}
				}
			}
		}
	}
}
