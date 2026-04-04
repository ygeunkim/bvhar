#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <bvhar/ols>

static Eigen::MatrixXd gen_ts(int num_data, int dim, unsigned int seed = 1) {
	BVHAR_BHRNG rng(seed);
	Eigen::MatrixXd time_series(num_data, dim);
	for (int i = 0; i < num_data * dim; ++i) {
		time_series(i) = baecon::bvhar::normal_rand(rng);
	}
	return time_series;
}

TEST_CASE("Response matrix", "[design]") {
	int num_data = 50;
	int dim = 3;
	int lag = 2;
	Eigen::MatrixXd y = gen_ts(num_data, dim);
	Eigen::MatrixXd y0 = baecon::bvhar::build_y0(y, lag, lag + 1);
	REQUIRE(y0.rows() == num_data - lag);
	REQUIRE(y0.cols() == dim);
}

TEST_CASE("Design matrix", "[design][var]") {
	int num_data = 50;
	int dim = 3;
	int lag = 2;
	Eigen::MatrixXd y = gen_ts(num_data, dim);

	SECTION("with intercept") {
		Eigen::MatrixXd x0 = baecon::bvhar::build_x0(y, lag, true);
		REQUIRE(x0.rows() == num_data - lag);
		REQUIRE(x0.cols() == dim * lag + 1);
		REQUIRE(x0.col(x0.cols() - 1).isOnes());
	}

	SECTION("without intercept") {
		Eigen::MatrixXd x0 = baecon::bvhar::build_x0(y, lag, false);
		REQUIRE(x0.rows() == num_data - lag);
		REQUIRE(x0.cols() == dim * lag);
	}
}

TEST_CASE("VHAR design", "[design][vhar]") {
	int dim = 3;
	int week = 5;
	int month = 22;

	SECTION("with intercept") {
		Eigen::MatrixXd har_trans = baecon::bvhar::build_vhar(dim, week, month, true);
		REQUIRE(har_trans.rows() == 3 * dim + 1);
		REQUIRE(har_trans.cols() == month * dim + 1);
	}

	SECTION("without intercept") {
		Eigen::MatrixXd har_trans = baecon::bvhar::build_vhar(dim, week, month, false);
		REQUIRE(har_trans.rows() == 3 * dim);
		REQUIRE(har_trans.cols() == month * dim);
	}
}

TEST_CASE("OLS methods", "[ols]") {
	int num_data = 50;
	int dim = 3;
	int lag = 2;
	Eigen::MatrixXd y = gen_ts(num_data, dim);
	Eigen::MatrixXd x0 = baecon::bvhar::build_x0(y, lag, true);
	Eigen::MatrixXd y0 = baecon::bvhar::build_y0(y, lag, lag + 1);

	baecon::bvhar::MultiOls ols_nor(x0, y0);
	baecon::bvhar::LltOls ols_llt(x0, y0);
	baecon::bvhar::QrOls ols_qr(x0, y0);

	Eigen::MatrixXd coef_nor = ols_nor.returnCoef();
	Eigen::MatrixXd coef_llt = ols_llt.returnCoef();
	Eigen::MatrixXd coef_qr = ols_qr.returnCoef();

	REQUIRE(coef_nor.isApprox(coef_llt, 1e-8));
	REQUIRE(coef_nor.isApprox(coef_qr, 1e-8));
}

TEST_CASE("OlsVar", "[ols][var]") {
	int num_data = 50;
	int dim = 3;
	int lag = 2;
	Eigen::MatrixXd y = gen_ts(num_data, dim);
	baecon::bvhar::OlsVar var(y, lag, true, 1);
	BVHAR_LIST res = var.returnOlsRes();

	REQUIRE(BVHAR_CONTAINS(res, "coefficients"));
	REQUIRE(BVHAR_CONTAINS(res, "fitted.values"));
	REQUIRE(BVHAR_CONTAINS(res, "residuals"));
	REQUIRE(BVHAR_CONTAINS(res, "covmat"));

	auto coef = BVHAR_CAST<Eigen::MatrixXd>(res["coefficients"]);
	auto yhat = BVHAR_CAST<Eigen::MatrixXd>(res["fitted.values"]);
	auto resid = BVHAR_CAST<Eigen::MatrixXd>(res["residuals"]);
	auto cov = BVHAR_CAST<Eigen::MatrixXd>(res["covmat"]);

	REQUIRE(coef.rows() == dim * lag + 1);
	REQUIRE(coef.cols() == dim);
	REQUIRE(yhat.rows() == num_data - lag);
	REQUIRE(yhat.cols() == dim);
	REQUIRE(resid.rows() == num_data - lag);
	REQUIRE(resid.cols() == dim);
	REQUIRE(cov.rows() == dim);
	REQUIRE(cov.cols() == dim);
}

TEST_CASE("OlsVhar", "[ols][vhar]") {
	int num_data = 50;
	int dim = 3;
	int week = 5;
	int month = 22;
	Eigen::MatrixXd y = gen_ts(num_data, dim);
	baecon::bvhar::OlsVhar vhar(y, week, month, true, 1);
	BVHAR_LIST res = vhar.returnOlsRes();

	REQUIRE(BVHAR_CONTAINS(res, "coefficients"));
	REQUIRE(BVHAR_CONTAINS(res, "HARtrans"));

	auto coef = BVHAR_CAST<Eigen::MatrixXd>(res["coefficients"]);
	auto har_trans = BVHAR_CAST<Eigen::MatrixXd>(res["HARtrans"]);

	REQUIRE(coef.rows() == 3 * dim + 1);
	REQUIRE(coef.cols() == dim);
	REQUIRE(har_trans.rows() == 3 * dim + 1);
	REQUIRE(har_trans.cols() == month * dim + 1);
}

TEST_CASE("OlsSimulator", "[ols][simulator]") {
	int num_data = 50;
	int dim = 3;
	int lag = 2;
	BVHAR_BHRNG rng(1);
	int dim_design = dim * lag + 1;
	Eigen::MatrixXd var_coef(dim_design, dim);
	for (int i = 0; i < dim_design * dim; ++i) {
		var_coef(i) = baecon::bvhar::normal_rand(rng) / 5;
	}
	Eigen::MatrixXd var_cov = Eigen::MatrixXd::Identity(dim, dim) / 3;
	auto dgp_run = std::make_unique<baecon::bvhar::OlsSimulator>(
		num_data, 0, lag, Eigen::MatrixXd::Zero(lag, dim),
		var_coef, var_cov, 2, 1
	);
	Eigen::MatrixXd ts = dgp_run->returnDgp();
	REQUIRE(ts.rows() == num_data);
	REQUIRE(ts.cols() == dim);
}
