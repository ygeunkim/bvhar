#ifndef MINNESOTA_H
#define MINNESOTA_H

// #include <RcppEigen.h>
#include "bvhardesign.h"
#include <memory> // std::unique_ptr

namespace bvhar {

struct MinnSpec {
	Eigen::VectorXd _sigma;
	double _lambda;
	double _eps;

	MinnSpec(Rcpp::List& bayes_spec);
};

struct BvarSpec : public MinnSpec {
	Eigen::VectorXd _delta;

	BvarSpec(Rcpp::List& bayes_spec);
};

struct BvharSpec : public MinnSpec {
	Eigen::VectorXd _daily;
	Eigen::VectorXd _weekly;
	Eigen::VectorXd _monthly;

	BvharSpec(Rcpp::List& bayes_spec);
};

class Minnesota {
public:
	Minnesota(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy);
	virtual ~Minnesota() = default;
	void estimateCoef();
	virtual void fitObs();
	void estimateCov(); // Posterior IW scale
	Rcpp::List returnMinnRes();
private:
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
	int dim;
	int num_design;
	int dim_design;
	int num_dummy; // kp + k(+ 1)
	int num_augment; // n + num_dummy
	Eigen::MatrixXd prior_prec;
	Eigen::MatrixXd prior_mean;
	Eigen::MatrixXd prior_scale;
	int prior_shape;
	Eigen::MatrixXd ystar; // [Y0, Yp]
	Eigen::MatrixXd xstar; // [X0, Xp]
	Eigen::MatrixXd coef; // MN mean
	Eigen::MatrixXd prec; // MN precision
	Eigen::MatrixXd yhat;
	Eigen::MatrixXd resid;
	Eigen::MatrixXd yhat_star;
	Eigen::MatrixXd scale; // IW scale
};

class MinnBvar {
public:
	MinnBvar(const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean);
	virtual ~MinnBvar() = default;
	Rcpp::List returnMinnRes();
private:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
};

class MinnBvhar {
public:
	MinnBvhar(const Eigen::MatrixXd& y, int week, int month, const MinnSpec& spec, const bool include_mean);
	virtual ~MinnBvhar() = default;
	virtual Rcpp::List returnMinnRes() = 0;
protected:
	int week;
	int month;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	Eigen::MatrixXd var_design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd har_trans;
	Eigen::MatrixXd design;
	Eigen::MatrixXd dummy_design;
};

class MinnBvharS : public MinnBvhar {
public:
	MinnBvharS(const Eigen::MatrixXd& y, int week, int month, const BvarSpec& spec, const bool include_mean);
	virtual ~MinnBvharS() = default;
	Rcpp::List returnMinnRes() override;
private:
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd dummy_response;
};

class MinnBvharL : public MinnBvhar {
public:
	MinnBvharL(const Eigen::MatrixXd& y, int week, int month, const BvharSpec& spec, const bool include_mean);
	virtual ~MinnBvharL() = default;
	Rcpp::List returnMinnRes() override;
private:
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd dummy_response;
};

} // namespace bvhar

#endif // MINNESOTA_H