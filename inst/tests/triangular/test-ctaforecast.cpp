#include "./includes.h"

namespace baecon {
namespace bvhar {
namespace tests {

std::vector<Eigen::MatrixXd> run_bvar_forecast(
	const Eigen::MatrixXd time_series, BVHAR_LIST_OF_LIST& mcmc_record,
	int num_chains, int nthreads, int lag, int n_ahead,
	int cov_type,
	bool include_mean,
	bool stable, bool sparse, double level,
	bool insample
) {
	BVHAR_LIST fit_record = baecon::bvhar::transpose_dict(mcmc_record);
	Eigen::VectorXi seed_chain = Eigen::VectorXi::Random(num_chains);
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::VectorXd>> {
		if (cov_type == 2) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, lag, n_ahead, time_series,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads//, true
			);
		}
		return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
			num_chains, lag, n_ahead, time_series,
			sparse, level, fit_record,
			seed_chain, include_mean, stable, nthreads
		);
	}();
	if (insample) {
		return forecaster->returnPredict();
	}
	return forecaster->returnForecast();
}

} // namespace tests
} // namespace bvhar
} // namespace baecon

TEST_CASE("Forecasting with BVAR: CTA", "[triangular][forecast]") {
	int num_chains = 1;
	int nthreads = 1;
	int num_iter = 5;
	int num_burn = 1;
	int thin = 2;
	int dim = 3;
	int num_data = 30;
	int lag = 2;
	bool include_mean = true;
	// int n_ahead = 3;
	double level = 0;
	bool ggl = true;
	int group_type = 2;
	int prior_type = 3;
	int contem_prior_type = 3;
	Eigen::MatrixXd time_series = baecon::bvhar::tests::gen_ts(num_data, dim, 1);
	for (int cov_type = 1; cov_type <= 2; ++cov_type) {
		BVHAR_LIST_OF_LIST cta_mcmc = baecon::bvhar::tests::run_bvar_cta(
			time_series,
			num_chains, nthreads, num_iter, num_burn, thin,
			dim, lag,
			group_type, cov_type, prior_type, contem_prior_type,
			include_mean, ggl
		);
		for (bool sparse : {true, false}) {
			for (bool stable : {false}) {
				for (bool insample : {true, false}) {
					DYNAMIC_SECTION(
						"cov_type=" << cov_type
							<< ", sparse=" << sparse
							<< ", stable=" << stable
							<< ", insample=" << insample
					) {
						int n_ahead = insample ? num_data - lag : 3;
						auto res = baecon::bvhar::tests::run_bvar_forecast(
							time_series, cta_mcmc,
							num_chains, nthreads, lag, n_ahead,
							cov_type, include_mean, stable, sparse, level, insample
						);
						REQUIRE(res.size() == num_chains);
					}
				}
			}
		}
	}
}
