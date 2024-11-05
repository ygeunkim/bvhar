#ifndef REGFORECASTER_H
#define REGFORECASTER_H

// #include "mcmcreg.h"
#include "bvharforecaster.h"

namespace bvhar {

class RegVarForecaster;
class RegVharForecaster;
class RegVarSelectForecaster;
class RegVharSelectForecaster;

class RegVarForecaster : public RegForecaster {
public:
	RegVarForecaster(const LdltRecords& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed)
	: RegForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05);
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~RegVarForecaster() = default;

protected:
	void computeMean() override {
		// post_mean = last_pvec.transpose() * coef_mat;
		post_mean = coef_mat.transpose() * last_pvec;
	}
};

class RegVharForecaster : public RegForecaster {
public:
	RegVharForecaster(const LdltRecords& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed)
	: RegForecaster(records, step, response_mat, month, include_mean, filter_stable, seed), har_trans(har_trans.sparseView()) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05, har_trans.topLeftCorner(3 * dim, month * dim).sparseView());
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~RegVharForecaster() = default;
	
protected:
	// Eigen::MatrixXd har_trans;
	Eigen::SparseMatrix<double> har_trans;
	void computeMean() override {
		// post_mean = last_pvec.transpose() * har_trans.transpose() * coef_mat;
		post_mean = coef_mat.transpose() * har_trans * last_pvec;
	}
};

class RegVarSelectForecaster : public RegVarForecaster {
public:
	RegVarSelectForecaster(const LdltRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed)
	: RegVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	RegVarSelectForecaster(const LdltRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed)
	: RegVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed), activity_graph(selection) {}
	virtual ~RegVarSelectForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

class RegVharSelectForecaster : public RegVharForecaster {
public:
	RegVharSelectForecaster(const LdltRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed)
	: RegVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	RegVharSelectForecaster(const LdltRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed)
	: RegVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed), activity_graph(selection) {}
	virtual ~RegVharSelectForecaster() = default;
	
protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

} // namespace bvhar

#endif // REGFORECASTER_H