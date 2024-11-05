#ifndef SVFORECASTER_H
#define SVFORECASTER_H

// #include "mcmcsv.h"
#include "bvharforecaster.h"

namespace bvhar {

using SvVarForecaster = McmcVarForecaster<SvForecaster>;
using SvVharForecaster = McmcVharForecaster<SvForecaster>;
class SvVarSelectForecaster;
class SvVharSelectForecaster;

class SvVarSelectForecaster : public SvVarForecaster {
public:
	SvVarSelectForecaster(const SvRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	SvVarSelectForecaster(const SvRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed), activity_graph(selection) {}
	virtual ~SvVarSelectForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

class SvVharSelectForecaster : public SvVharForecaster {
public:
	SvVharSelectForecaster(const SvRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, sv, seed),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	SvVharSelectForecaster(const SvRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, sv, seed), activity_graph(selection) {}
	virtual ~SvVharSelectForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

} // namespace bvhar

#endif // SVFORECASTER_H