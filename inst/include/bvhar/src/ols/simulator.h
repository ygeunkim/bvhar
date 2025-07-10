#ifndef BVHAR_OLS_SIMULATOR_H
#define BVHAR_OLS_SIMULATOR_H

#include "./forecaster.h"

namespace bvhar {

class OlsSimulator;

inline std::unique_ptr<OlsErrorGenerator> initialize_olsgenerator(
	int num_iter, int num_burn, const Eigen::MatrixXd& init,
	const Eigen::MatrixXd sig_error,
	int method, unsigned int seed,
	Optional<int> mvt_df = NULLOPT
) {
	std::unique_ptr<OlsErrorGenerator> dgp_ptr;
	Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(sig_error.cols());
	if (mvt_df) {
		return std::make_unique<OlsStudentGenerator>(sig_mean, sig_error, *mvt_df, method, seed);
	}
	return std::make_unique<OlsGaussianGenerator>(sig_mean, sig_error, method, seed);
}

class OlsSimulator : public MultistepForecastRun<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsSimulator(
		int num_iter, int num_burn, int lag,
		const Eigen::MatrixXd& init, const Eigen::MatrixXd& coef_mat,
		const Eigen::MatrixXd sig_error,
		int method, unsigned int seed,
		Optional<int> mvt_df = NULLOPT
	) : num_iter(num_iter) {
		auto dgp_updater = initialize_olsgenerator(num_iter, num_burn, init, sig_error, method, seed, mvt_df);
		OlsFit ols_fit(coef_mat, lag);
		bool include_mean = coef_mat.rows() == sig_error.cols() * lag + 1;
		generator = std::make_unique<VarForecaster>(ols_fit, num_iter + num_burn, init, include_mean, std::move(dgp_updater));
	}

	OlsSimulator(
		int num_iter, int num_burn, int week, int month,
		const Eigen::MatrixXd& init, const Eigen::MatrixXd& coef_mat,
		const Eigen::MatrixXd sig_error,
		int method, unsigned int seed,
		Optional<int> mvt_df = NULLOPT
	) : num_iter(num_iter) {
		auto dgp_updater = initialize_olsgenerator(num_iter, num_burn, init, sig_error, method, seed, mvt_df);
		OlsFit ols_fit(coef_mat, month);
		bool include_mean = coef_mat.rows() == sig_error.cols() * 3 + 1;
		Eigen::MatrixXd har_trans = build_vhar(init.cols(), week, month, include_mean);
		generator = std::make_unique<VharForecaster>(ols_fit, num_iter + num_burn, init, har_trans, include_mean, std::move(dgp_updater));
	}

	virtual ~OlsSimulator() = default;
	
	Eigen::MatrixXd returnDgp() {
		return generator->doForecast().bottomRows(num_iter);
	}

private:
	int num_iter;
	std::unique_ptr<OlsForecaster> generator;
};

} // namespace bvhar

#endif // BVHAR_OLS_SIMULATOR_H