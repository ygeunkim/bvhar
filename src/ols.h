#ifndef OLS_H
#define OLS_H

#include <RcppEigen.h>

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
	// OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean);
	OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean, int method);
	virtual ~OlsVar() = default;
	virtual void estimateCoef();
	virtual void fitObs();
	void estimateCov();
	// Rcpp::List returnOlsRes();
protected:
	int lag;
	bool const_term;
	int dim; // k
	Eigen::MatrixXd data;
	std::unique_ptr<MultiOls> _ols; // _ols(new MultiOls(...))
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
	void estimateCoef() override;
private:
	Eigen::LLT<Eigen::MatrixXd> llt_selfadjoint;
};

class QrVar : public OlsVar {
public:
	QrVar(const Eigen::MatrixXd& y, int lag, const bool include_mean);
	virtual ~QrVar() = default;
	void estimateCoef() override;
private:
	Eigen::HouseholderQR<Eigen::MatrixXd> qr_design;
};

#endif