#ifndef BVHARFORECASTER_H
#define BVHARFORECASTER_H

#include "bvharmcmc.h"

namespace bvhar {

class McmcForecaster;
class McmcRegForecaster;
class McmcSvForecaster;
template <typename BaseForecaster, typename Record> class McmcVarForecaster;
template <typename BaseForecaster, typename Record> class McmcVharForecaster;

class McmcForecaster {
public:
	McmcForecaster(RegRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, unsigned int seed)
	: reg_record(records), rng(seed),
		response(response_mat), include_mean(include_mean),
		step(step), dim(response.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_coef(reg_record.coef_record.cols()),
		num_alpha(include_mean ? num_coef - dim : num_coef), nrow_coef(num_alpha / dim),
		num_sim(reg_record.coef_record.rows()),
		last_pvec(Eigen::VectorXd::Zero(dim_design)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		post_mean(Eigen::VectorXd::Zero(dim)),
		predictive_distn(Eigen::MatrixXd::Zero(step, num_sim * dim)),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		standard_normal(Eigen::VectorXd::Zero(dim)),
		lpl(Eigen::VectorXd::Zero(step)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(var_lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		post_mean = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~McmcForecaster() = default;
	Eigen::MatrixXd forecastDensity() {
		std::lock_guard<std::mutex> lock(mtx);
		for (int h = 0; h < step; h++) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			for (int i = 0; i < num_sim; i++) {
				forecast(i);
				predictive_distn.block(h, i * dim, 1, dim) = post_mean.transpose();
			}
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
		return predictive_distn;
	}
	Eigen::MatrixXd forecastDensity(const Eigen::VectorXd& valid_vec) {
		std::lock_guard<std::mutex> lock(mtx);
		for (int h = 0; h < step; h++) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			for (int i = 0; i < num_sim; i++) {
				forecast(i);
				predictive_distn.block(h, i * dim, 1, dim) = post_mean.transpose();
				updateLpl(h, valid_vec);
			}
			lpl[h] /= num_sim;
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
		return predictive_distn;
	}
	Eigen::VectorXd returnLplRecord() {
		return lpl;
	}
	double returnLpl() {
		return lpl.mean();
	}

protected:
	RegRecords reg_record;
	boost::random::mt19937 rng;
	std::mutex mtx;
	Eigen::MatrixXd response; // y0
	bool include_mean;
	int step;
	int dim;
	int var_lag; // VAR order or month order of VHAR
	int dim_design;
	int num_coef;
	int num_alpha;
	int nrow_coef; // dim_design in VAR and dim_har in VHAR (without constant term)
	int num_sim;
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd sv_update; // d_1, ..., d_m
	Eigen::VectorXd post_mean; // posterior mean and y_(T + h - 1)
	Eigen::MatrixXd predictive_distn; // rbind(step), cbind(sims)
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::VectorXd standard_normal; // Z ~ N(0, I)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
	Eigen::VectorXd lpl; // average log-predictive likelihood
	virtual void computeMean(int i) = 0;
	virtual void updateCov(int i) = 0;
	virtual void generateInnov() = 0;
	void updateVariance(int i) {
		updateCov(i);
		for (int j = 0; j < dim; j++) {
			standard_normal[j] = normal_rand(rng);
		}
		generateInnov(); // D^(1/2) Z ~ N(0, D)
	}
	virtual void updateLpl(int h, const Eigen::VectorXd& valid_vec) = 0;

private:
	void forecast(int i) {
		coef_mat.topRows(nrow_coef) = unvectorize(reg_record.coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows(1) = reg_record.coef_record.row(i).tail(dim);
		}
		last_pvec.head(dim) = post_mean;
		computeMean(i);
		updateVariance(i);
		post_mean += contem_mat.triangularView<Eigen::UnitLower>().solve(standard_normal); // N(post_mean, L^-1 D L)
	}
};

class McmcRegForecaster : public McmcForecaster {
public:
	McmcRegForecaster(LdltRecords2& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, unsigned int seed)
	: McmcForecaster(records, step, response_mat, ord, include_mean, seed),
		fac_record(records.fac_record) {}
	virtual ~McmcRegForecaster() = default;

protected:
	void computeMean(int i) override {}
	void updateCov(int i) override {
		sv_update = fac_record.row(i).transpose();
		contem_mat = build_inv_lower(dim, reg_record.contem_coef_record.row(i));
	}
	void generateInnov() override {
		standard_normal.array() *= sv_update.cwiseSqrt().array();
	}
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) override {
		lpl[h] += sv_update.sum() / 2 - dim * log(2 * M_PI) / 2 - (sv_update.cwiseSqrt().cwiseInverse().array() * (contem_mat * (post_mean - valid_vec)).array()).matrix().squaredNorm() / 2;
	}

private:
	Eigen::MatrixXd fac_record;
};

class McmcSvForecaster : public McmcForecaster {
public:
	McmcSvForecaster(SvRecords2& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, unsigned int seed)
	: McmcForecaster(records, step, response_mat, ord, include_mean, seed),
		lvol_sig_record(records.lvol_sig_record),
		h_last_record(records.lvol_record.rightCols(dim)) {}
	virtual ~McmcSvForecaster() = default;

protected:
	void computeMean(int i) override {}
	void updateCov(int i) override {
		for (int j = 0; j < dim; ++j) {
			standard_normal[j] = normal_rand(rng);
		}
		standard_normal.array() *= lvol_sig_record.row(i).cwiseSqrt().array(); // sig_h Z ~ N(0, sig_h^2)
		sv_update = h_last_record.row(i).transpose() + standard_normal; // h_(t+1) = h_t + u_t
		h_last_record.row(i) = sv_update; // for next update
	}
	void generateInnov() override {
		standard_normal.array() *= sv_update.cwiseSqrt().array();
	}
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) override {
		lpl[h] += sv_update.sum() / 2 - dim * log(2 * M_PI) / 2 - ((-sv_update / 2).array().exp() * (contem_mat * (post_mean - valid_vec)).array()).matrix().squaredNorm() / 2;
	}

private:
	Eigen::MatrixXd lvol_sig_record;
	Eigen::MatrixXd h_last_record; // h_T record
};

template <
	typename BaseForecaster = McmcRegForecaster,
	typename Record = typename std::conditional<
		std::is_same<BaseForecaster, McmcRegForecaster>::value,
		LdltRecords2,
		SvRecords2
	>::type
>
class McmcVarForecaster : public BaseForecaster {
public:
	McmcVarForecaster(Record& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, unsigned int seed)
	: BaseForecaster(records, step, response_mat, lag, include_mean, seed) {}
	virtual ~McmcVarForecaster() = default;

protected:
	using BaseForecaster::post_mean;
	using BaseForecaster::last_pvec;
	using BaseForecaster::coef_mat;
	void computeMean(int i) override {
		post_mean = last_pvec.transpose() * coef_mat;
	}
};

template <
	typename BaseForecaster = McmcRegForecaster,
	typename Record = typename std::conditional<
		std::is_same<BaseForecaster, McmcRegForecaster>::value,
		LdltRecords2,
		SvRecords2
	>::type
>
class McmcVharForecaster : public BaseForecaster {
public:
	McmcVharForecaster(Record& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, unsigned int seed)
	: BaseForecaster(records, step, response_mat, month, include_mean, seed),
		har_trans(har_trans) {}
	virtual ~McmcVharForecaster() = default;

protected:
	using BaseForecaster::post_mean;
	using BaseForecaster::last_pvec;
	using BaseForecaster::coef_mat;
	void computeMean(int i) override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * coef_mat;
	}

private:
	Eigen::MatrixXd har_trans;
};

}; // namespace bvhar

#endif // BVHARFORECASTER_H