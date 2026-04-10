#include "./includes.h"

TEST_CASE("BVAR: Corrected Triangular Algorithm", "[triangular][mcmc]") {
	int num_chains = 1;
	int nthreads = 1;
	int num_iter = 5;
	int num_burn = 1;
	int thin = 2;
	int dim = 3;
	int num_data = 30;
	int lag = 2;
	bool include_mean = true;
	Eigen::MatrixXd time_series = baecon::bvhar::tests::gen_ts(num_data, dim, 1);
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
							BVHAR_LIST_OF_LIST res = baecon::bvhar::tests::run_bvar_cta(
								time_series,
								num_chains, nthreads, num_iter, num_burn, thin,
								dim, lag,
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
