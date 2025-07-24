#ifndef BVHAR_OLS_FORECASTER_H
#define BVHAR_OLS_FORECASTER_H

// #include "../core/common.h"
#include "../core/forecaster.h"
#include "../math/random.h"
#include "./ols.h"
#include <type_traits>

namespace bvhar {

class OlsExogenForecaster;
class OlsErrorGenerator;
class OlsGaussianGenerator;
class OlsStudentGenerator;
class OlsForecaster;
class VarForecaster;
class VharForecaster;
class OlsForecastRun;
class OlsOutforecastRun;
class OlsRollforecastRun;
class OlsExpandforecastRun;
template <typename BaseOutForecast> class VarOutforecastRun;
template <typename BaseOutForecast> class VharOutforecastRun;

class OlsExogenForecaster : public ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsExogenForecaster() {}
	OlsExogenForecaster(int lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef)
	: ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd>(lag, exogen),
		coef_mat(exogen_coef) {
		last_pvec = vectorize_eigen(exogen.topRows(lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
	}
	virtual ~OlsExogenForecaster() = default;

	void appendForecast(Eigen::VectorXd& point_forecast, const int h) override {
		last_pvec = vectorize_eigen(exogen.middleRows(h, lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
		// point_forecast += last_pvec.transpose() * coef_mat;
		point_forecast += coef_mat.transpose() * last_pvec;
	}

private:
	// int dim_exogen_design;
	Eigen::MatrixXd coef_mat;
};

class OlsErrorGenerator : public AutoregGenerator<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	// OlsErrorGenerator() {}
	OlsErrorGenerator(int dim, unsigned int seed)
	: AutoregGenerator<Eigen::MatrixXd, Eigen::VectorXd>(seed),
		dim(dim) {
		// error_term = Eigen::MatrixXd::Zero(num_iter, dim);
		error_term = Eigen::VectorXd::Zero(dim);
	}
	virtual ~OlsErrorGenerator() = default;

protected:
	int dim;
	// Eigen::VectorXd error_mean;
	// Eigen::MatrixXd error_sig;
};

class OlsGaussianGenerator : public OlsErrorGenerator {
public:
	OlsGaussianGenerator(const Eigen::VectorXd& error_mean, const Eigen::MatrixXd& error_sig, int method, unsigned int seed)
	: OlsErrorGenerator(error_sig.cols(), seed),
		alg_type(method),
		error_mean(error_mean), error_sig(error_sig) {}
	virtual ~OlsGaussianGenerator() = default;

	void appendError(Eigen::VectorXd& point_forecast) override {
		switch (alg_type) {
			case 1: {
				error_term = sim_mgaussian_eigen(1, error_mean, error_sig, rng).transpose();
				break;
			}
			case 2: {
				error_term = sim_mgaussian_chol(1, error_mean, error_sig, rng).transpose();
				break;
			}
		}
		point_forecast.array() += error_term.array();
	}

private:
	int alg_type;
	Eigen::VectorXd error_mean;
	Eigen::MatrixXd error_sig;
};

class OlsStudentGenerator : public OlsErrorGenerator {
public:
	OlsStudentGenerator(const Eigen::VectorXd& error_mean, const Eigen::MatrixXd& error_sig, double mvt_df, int method, unsigned int seed)
	: OlsErrorGenerator(error_sig.cols(), seed),
		alg_type(method),
		mvt_df(mvt_df), error_mean(error_mean), error_sig(error_sig) {}
	virtual ~OlsStudentGenerator() = default;

	void appendError(Eigen::VectorXd& point_forecast) override {
		switch (alg_type) {
			case 1: {
				error_term = sim_mstudent_eigen(1, mvt_df, error_mean, error_sig * (mvt_df - 2) / mvt_df, rng).transpose();
				break;
			}
			case 2: {
				error_term = sim_mstudent_chol(1, mvt_df, error_mean, error_sig * (mvt_df - 2) / mvt_df, rng).transpose();
				break;
			}
		}
		point_forecast.array() += error_term.array();
	}

private:
	int alg_type;
	double mvt_df;
	Eigen::VectorXd error_mean;
	Eigen::MatrixXd error_sig;
};

class OlsForecaster : public MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecaster(
		const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, fit._ord),
		coef_mat(fit._coef), include_mean(include_mean), dim(coef_mat.cols()),
		dim_design(include_mean ? lag * dim + 1 : lag * dim) {
		initLagged();
		if (dgp_updater) {
			error_updater = std::move(*dgp_updater);
		}
	}
	OlsForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster> exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, fit._ord),
		exogen_updater(std::move(exogen_updater)),
		coef_mat(fit._coef), include_mean(include_mean), dim(coef_mat.cols()),
		dim_design(include_mean ? lag * dim + 1 : lag * dim) {
		initLagged();
		if (dgp_updater) {
			error_updater = std::move(*dgp_updater);
		}
	}
	virtual ~OlsForecaster() = default;
	Eigen::MatrixXd forecastPoint() {
		return this->doForecast();
	}

	Eigen::VectorXd getLastForecast() override {
		return this->doForecast().template bottomRows<1>();
	}

protected:
	std::unique_ptr<OlsExogenForecaster> exogen_updater;
	std::unique_ptr<OlsErrorGenerator> error_updater;
	Eigen::MatrixXd coef_mat;
	bool include_mean;
	int dim;
	int dim_design;

	void initLagged() override {
		pred_save = Eigen::MatrixXd::Zero(step, dim);
		last_pvec = Eigen::VectorXd::Zero(dim_design);
		last_pvec[dim_design - 1] = 1.0;
		last_pvec.head(lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		tmp_vec = last_pvec.segment(dim, (lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
		point_forecast = last_pvec.head(dim); // y_T
	}

	void setRecursion() override {
		last_pvec.segment(dim, (lag - 1) * dim) = tmp_vec;
		last_pvec.head(dim) = point_forecast;
	}

	void updateRecursion() override {
		tmp_vec = last_pvec.head((lag - 1) * dim);
	}

	void updatePred(const int h, const int i) override {
		computeMean();
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		if (error_updater) {
			error_updater->appendError(point_forecast);
		}
		pred_save.row(h) = point_forecast.transpose();
	}

	virtual void computeMean() = 0;
};

class VarForecaster : public OlsForecaster {
public:
	VarForecaster(
		const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: OlsForecaster(fit, step, response_mat, include_mean, std::move(dgp_updater)) {}
	VarForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster> exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: OlsForecaster(fit, std::move(exogen_updater), step, response_mat, include_mean, std::move(dgp_updater)) {}
	virtual ~VarForecaster() = default;

protected:
	void computeMean() override {
		point_forecast = last_pvec.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * Ahat
	}
};

class VharForecaster : public OlsForecaster {
public:
	VharForecaster(
		const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: OlsForecaster(fit, step, response_mat, include_mean, std::move(dgp_updater)), har_trans(har_trans) {}
	VharForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster> exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean,
		BVHAR_OPTIONAL<std::unique_ptr<OlsErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: OlsForecaster(fit, std::move(exogen_updater), step, response_mat, include_mean, std::move(dgp_updater)), har_trans(har_trans) {}
	virtual ~VharForecaster() = default;

protected:
	void computeMean() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * C(HAR) * Ahat
	}

private:
	Eigen::MatrixXd har_trans;
};

class OlsForecastRun : public MultistepForecastRun<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecastRun(int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean) {
		bvhar::OlsFit ols_fit(coef_mat, lag);
		forecaster = std::make_unique<VarForecaster>(ols_fit, step, response_mat, include_mean);
	}
	OlsForecastRun(
		int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		int exogen_lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	) {
		OlsFit ols_fit(coef_mat, lag);
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(exogen_lag, exogen, exogen_coef);
		forecaster = std::make_unique<VarForecaster>(ols_fit, std::move(exogen_updater), step, response_mat, include_mean);
	}
	OlsForecastRun(int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		OlsFit ols_fit(coef_mat, month);
		forecaster = std::make_unique<VharForecaster>(ols_fit, step, response_mat, har_trans, include_mean);
	}
	OlsForecastRun(
		int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		int exogen_lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		OlsFit ols_fit(coef_mat, month);
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(exogen_lag, exogen, exogen_coef);
		forecaster = std::make_unique<VharForecaster>(ols_fit, std::move(exogen_updater), step, response_mat, har_trans, include_mean);
	}
	virtual ~OlsForecastRun() = default;
	
	Eigen::MatrixXd returnForecast() {
		return forecaster->doForecast();
	}

protected:
	std::unique_ptr<OlsForecaster> forecaster;
};

class OlsOutforecastRun {
public:
	OlsOutforecastRun(
		const Eigen::MatrixXd& y, int lag,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		int method, int nthreads,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: dim(y.cols()), num_window(y.rows()), num_test(y_test.rows()), num_horizon(num_test - step + 1), step(step),
		lag(lag), nthreads(nthreads), include_mean(include_mean),
		roll_mat(num_horizon), roll_y0(num_horizon), y_test(y_test),
		model(num_horizon), forecaster(num_horizon), out_forecast(num_horizon),
		roll_exogen_mat(num_horizon), roll_exogen(num_horizon), lag_exogen(exogen_lag) {}
	virtual ~OlsOutforecastRun() = default;
	
	void forecast() {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; ++window) {
			// OlsFit ols_fit = model[window]->returnOlsFit(lag);
			std::unique_ptr<OlsFit> ols_fit;
			Eigen::MatrixXd coef_mat = model[window]->returnCoef();
			if (lag_exogen) {
				int nrow_exogen = (*lag_exogen + 1) * roll_exogen_mat[window]->cols();
				ols_fit = std::make_unique<OlsFit>(coef_mat.topRows(coef_mat.rows() - nrow_exogen), lag);
				Eigen::MatrixXd exogen_coef = coef_mat.bottomRows(nrow_exogen);
				updateForecaster(*ols_fit, window, exogen_coef);
			} else {
				ols_fit = std::make_unique<OlsFit>(coef_mat, lag);
				updateForecaster(*ols_fit, window);
			}
			// out_forecast[window] = forecaster[window]->getLastForecast();
			out_forecast[window] = forecaster[window]->doForecast().bottomRows(1);
			model[window].reset();
			forecaster[window].reset();
		}
	}

	Eigen::MatrixXd returnForecast() {
		forecast();
		return std::accumulate(
			out_forecast.begin() + 1, out_forecast.end(), out_forecast[0],
			[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
				Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
				concat_mat << acc,
											curr;
				return concat_mat;
			}
		);
	}

protected:
	int dim, num_window, num_test, num_horizon, step, lag, nthreads;
	bool include_mean;
	std::vector<Eigen::MatrixXd> roll_mat;
	std::vector<Eigen::MatrixXd> roll_y0;
	Eigen::MatrixXd y_test;
	std::vector<std::unique_ptr<MultiOls>> model;
	std::vector<std::unique_ptr<OlsForecaster>> forecaster;
	std::vector<Eigen::MatrixXd> out_forecast;
	std::vector<BVHAR_OPTIONAL<Eigen::MatrixXd>> roll_exogen_mat;
	std::vector<BVHAR_OPTIONAL<Eigen::MatrixXd>> roll_exogen;
	BVHAR_OPTIONAL<int> lag_exogen;

	void initOls(int method) {
		for (int window = 0; window < num_horizon; ++window) {
			Eigen::MatrixXd design = buildDesign(window);
			model[window] = initialize_ols(design, roll_y0[window], method);
		}
	}

	virtual void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) = 0;

	void initialize(const Eigen::MatrixXd& y, int method, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) {
		initData(y, exogen);
		initOls(method);
	}

	virtual Eigen::MatrixXd buildDesign(int window) = 0;

	virtual void updateForecaster(const OlsFit& fit, int window) = 0;
	virtual void updateForecaster(const OlsFit& fit, int window, const Eigen::MatrixXd& exogen_coef) = 0;
};

class OlsRollforecastRun : public OlsOutforecastRun {
public:
	OlsRollforecastRun(
		const Eigen::MatrixXd& y, int lag,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		int method, int nthreads,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: OlsOutforecastRun(y, lag, include_mean, step, y_test, method, nthreads, exogen, exogen_lag) {}
	virtual ~OlsRollforecastRun() = default;

protected:
	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.middleRows(i, num_window);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
		if (lag_exogen) {
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).middleRows(i, num_window);
				roll_exogen[i] = (*exogen).middleRows(num_window - *lag_exogen + i, *lag_exogen + step);
			}
		}
	}
};

class OlsExpandforecastRun : public OlsOutforecastRun {
public:
	OlsExpandforecastRun(
		const Eigen::MatrixXd& y, int lag,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		int method, int nthreads,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: OlsOutforecastRun(y, lag, include_mean, step, y_test, method, nthreads, exogen, exogen_lag) {}
	virtual ~OlsExpandforecastRun() = default;
	
protected:
	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows(num_window + i);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
		if (lag_exogen && exogen) {
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).topRows(num_window + i);
				roll_exogen[i] = (*exogen).middleRows(num_window - *lag_exogen + i, *lag_exogen + step);
			}
		}
	}
};

template <typename BaseOutForecast = OlsRollforecastRun>
class VarOutforecastRun : public BaseOutForecast {
public:
	VarOutforecastRun(
		const Eigen::MatrixXd& y, int lag,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		int method, int nthreads,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: BaseOutForecast(y, lag, include_mean, step, y_test, method, nthreads, exogen, exogen_lag) {
		initialize(y, method, exogen);
	}
	virtual ~VarOutforecastRun() = default;

protected:
	using BaseOutForecast::step;
	using BaseOutForecast::dim;
	using BaseOutForecast::num_window;
	using BaseOutForecast::num_horizon;
	using BaseOutForecast::y_test;
	using BaseOutForecast::lag;
	using BaseOutForecast::include_mean;
	using BaseOutForecast::roll_mat;
	using BaseOutForecast::roll_y0;
	using BaseOutForecast::roll_exogen_mat;
	using BaseOutForecast::roll_exogen;
	using BaseOutForecast::lag_exogen;
	using BaseOutForecast::forecaster;
	using BaseOutForecast::initialize;
	using BaseOutForecast::initData;
	using BaseOutForecast::initOls;

	Eigen::MatrixXd buildDesign(int window) override {
		if (lag_exogen) {
			return build_x0(roll_mat[window], *(roll_exogen_mat[window]), lag, *lag_exogen, include_mean);
		}
		return build_x0(roll_mat[window], lag, include_mean);
	}

	void updateForecaster(const OlsFit& fit, int window) override {
		forecaster[window] = std::make_unique<VarForecaster>(fit, step, roll_mat[window], include_mean);
	}

	void updateForecaster(const OlsFit& fit, int window, const Eigen::MatrixXd& exogen_coef) override {
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(*lag_exogen, *(roll_exogen[window]), exogen_coef);
		forecaster[window] = std::make_unique<VarForecaster>(fit, std::move(exogen_updater), step, roll_mat[window], include_mean);
	}
};

template <typename BaseOutForecast = OlsRollforecastRun>
class VharOutforecastRun : public BaseOutForecast {
public:
	VharOutforecastRun(
		const Eigen::MatrixXd& y, int week, int month,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		int method, int nthreads,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: BaseOutForecast(y, month, include_mean, step, y_test, method, nthreads, exogen, exogen_lag),
		har_trans(build_vhar(dim, week, month, include_mean)) {
		initialize(y, method, exogen);
	}
	virtual ~VharOutforecastRun() = default;

protected:
	using BaseOutForecast::step;
	using BaseOutForecast::dim;
	using BaseOutForecast::num_window;
	using BaseOutForecast::num_horizon;
	using BaseOutForecast::y_test;
	using BaseOutForecast::lag;
	using BaseOutForecast::include_mean;
	using BaseOutForecast::roll_mat;
	using BaseOutForecast::roll_y0;
	using BaseOutForecast::roll_exogen_mat;
	using BaseOutForecast::roll_exogen;
	using BaseOutForecast::lag_exogen;
	using BaseOutForecast::forecaster;
	using BaseOutForecast::initialize;
	using BaseOutForecast::initData;
	using BaseOutForecast::initOls;

	Eigen::MatrixXd buildDesign(int window) override {
		if (lag_exogen) {
			int dim_design = include_mean ? lag * dim + 1 : lag * dim;
			int dim_har = include_mean ? 3 * dim + 1 : 3 * dim;
			int dim_exogen = (*lag_exogen + 1) * roll_exogen_mat[window]->cols();
			Eigen::MatrixXd vhar_design(roll_y0[window].rows(), dim_har + dim_exogen);
			Eigen::MatrixXd var_design = build_x0(roll_mat[window], *(roll_exogen_mat[window]), lag, *lag_exogen, include_mean);
			vhar_design.leftCols(dim_har) = var_design.leftCols(dim_design) * har_trans.transpose();
			vhar_design.rightCols(dim_exogen) = var_design.rightCols(dim_exogen);
			return vhar_design;
		}
		return build_x0(roll_mat[window], lag, include_mean) * har_trans.transpose();
	}

	void updateForecaster(const OlsFit& fit, int window) override {
		forecaster[window] = std::make_unique<VharForecaster>(fit, step, roll_mat[window], har_trans, include_mean);
	}

	void updateForecaster(const OlsFit& fit, int window, const Eigen::MatrixXd& exogen_coef) override {
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(*lag_exogen, *(roll_exogen[window]), exogen_coef);
		forecaster[window] = std::make_unique<VharForecaster>(fit, std::move(exogen_updater), step, roll_mat[window], har_trans, include_mean);
	}

private:
	Eigen::MatrixXd har_trans;
};

} // namespace bvhar

#endif // BVHAR_OLS_FORECASTER_H