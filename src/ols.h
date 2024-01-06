#ifndef OLS_H
#define OLS_H

#include <RcppEigen.h>

class OlsVar {
public:
	OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean);
	virtual ~OlsVar() = default;
	void estimateCoef();
	void fitObs();
	void estimateCov();
	Rcpp::List returnOlsRes();
protected:
	int lag;
	bool const_term;
	int dim; // k
	Eigen::MatrixXd data;
	Eigen::MatrixXd response;
	Eigen::MatrixXd design;
	int num_design; // n
	int dim_design; // kp( + 1)
	Eigen::MatrixXd coef;
	Eigen::MatrixXd yhat;
	Eigen::MatrixXd resid;
	Eigen::MatrixXd cov;
};

class LltVar : public OlsVar {
public:
	LltVar(const Eigen::MatrixXd& y, int lag, const bool include_mean);
	virtual ~LltVar() = default;
	void estimateCoef();
};

class QrVar : public OlsVar {
public:
	QrVar(const Eigen::MatrixXd& y, int lag, const bool include_mean);
	virtual ~QrVar() = default;
	void estimateCoef();
	void fitObs();
private:
	Eigen::HouseholderQR<Eigen::MatrixXd> qr_design;
	Eigen::MatrixXd q_mat;
};

#endif