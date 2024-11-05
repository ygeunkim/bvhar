#ifndef SVFORECASTER_H
#define SVFORECASTER_H

// #include "mcmcsv.h"
#include "bvharforecaster.h"

namespace bvhar {

class SvVarForecaster;
class SvVharForecaster;
class SvVarSelectForecaster;
class SvVharSelectForecaster;

class SvVarForecaster : public SvForecaster {
public:
	SvVarForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05);
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~SvVarForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * coef_mat;
	}
};

class SvVharForecaster : public SvForecaster {
public:
	SvVharForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvForecaster(records, step, response_mat, month, include_mean, filter_stable, sv, seed), har_trans(har_trans.sparseView()) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05, har_trans.topLeftCorner(3 * dim, month * dim).sparseView());
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~SvVharForecaster() = default;

protected:
	// Eigen::MatrixXd har_trans;
	Eigen::SparseMatrix<double> har_trans;
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * coef_mat;
	}
};

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