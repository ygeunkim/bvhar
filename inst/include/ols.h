#ifndef OLS_H
#define OLS_H

#include "bvhardesign.h"
#include <memory> // std::unique_ptr in source file

class MultiOls {
public:
	MultiOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
	virtual ~MultiOls() = default;
	virtual void estimateCoef();
	virtual void fitObs();
	void estimateCov();
	Rcpp::List returnOlsRes();
protected:
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	int dim; // k
	int num_design; // n
	int dim_design; // kp( + 1)
	Eigen::MatrixXd coef;
	Eigen::MatrixXd yhat;
	Eigen::MatrixXd resid;
	Eigen::MatrixXd cov;
};

class LltOls : public MultiOls {
public:
	LltOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
	virtual ~LltOls() = default;
	void estimateCoef() override;
private:
	Eigen::LLT<Eigen::MatrixXd> llt_selfadjoint;
};

class QrOls : public MultiOls {
public:
	QrOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
	virtual ~QrOls() = default;
	void estimateCoef() override;
private:
	Eigen::HouseholderQR<Eigen::MatrixXd> qr_design;
};

class OlsVar {
public:
	OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean, int method);
	virtual ~OlsVar() = default;
	Rcpp::List returnOlsRes();
protected:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	std::unique_ptr<MultiOls> _ols;
	Eigen::MatrixXd response;
	Eigen::MatrixXd design;
};

class OlsVhar {
public:
	OlsVhar(const Eigen::MatrixXd& y, int week, int month, const bool include_mean, int method);
	virtual ~OlsVhar() = default;
	Rcpp::List returnOlsRes();
protected:
	int week;
	int month;
	bool const_term;
	Eigen::MatrixXd data;
	std::unique_ptr<MultiOls> _ols;
	Eigen::MatrixXd response;
	Eigen::MatrixXd var_design;
	Eigen::MatrixXd design;
	Eigen::MatrixXd har_trans;
};

#endif