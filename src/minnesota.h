#ifndef MINNESOTA_H
#define MINNESOTA_H

#include <RcppEigen.h>
#include "bvhardesign.h"
#include "bvhardraw.h"
#include <memory> // std::unique_ptr

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

struct HierMinnSpec {
	double acc_scale;
	Eigen::MatrixXd obs_information;
	double gamma_shp;
	double gamma_rate;
	double invgam_shp;
	double invgam_scl;

	HierMinnSpec(Rcpp::List& bayes_spec);
};

class Minnesota {
public:
	Minnesota(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy);
	virtual ~Minnesota() = default;
	void estimateCoef();
	void fitObs();
	void estimateCov(); // Posterior IW scale
	Rcpp::List returnMinnRes();
protected:
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
	int shape;
};

class HierMinn : public Minnesota {
public:
	HierMinn(
		int num_iter,
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy,
		const HierMinnSpec& spec, const MinnSpec& init
	);
	virtual ~HierMinn() = default;
	void updateHyper(); // hyperparams using MH
	void updateMniw(); // coef and sigma
	void addStep();
	void doPosteriorDraws();
	void estimatePosterior();
	Rcpp::List returnRecords(int num_burn) const;
	Rcpp::List returnMinnRes(int num_burn);
private:
	int num_iter;
	int mcmc_step;
	bool is_accept;
	typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
	VectorXb accept_record;
	double numerator;
	double denom;
	double gamma_shp;
	double gamma_rate;
	double invgam_shp;
	double invgam_scl;
	Eigen::MatrixXd gaussian_variance;
	double lambda;
	Eigen::VectorXd psi;
	Eigen::VectorXd candprior;
	Eigen::VectorXd prevprior;
	Rcpp::List coef_and_sig;
	Eigen::VectorXd lam_record;
	Eigen::MatrixXd psi_record;
	Eigen::MatrixXd coef_record;
	Eigen::MatrixXd sig_record;
};

class MinnBvar {
public:
	MinnBvar(const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean);
	virtual ~MinnBvar() = default;
	Rcpp::List returnMinnRes();
protected:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
private:
	std::unique_ptr<Minnesota> _mn;
};

class HierBvar : public MinnBvar {
public:
	HierBvar(int num_iter, const Eigen::MatrixXd& y, int lag, const HierMinnSpec& spec, const BvarSpec& init, const bool include_mean);
	virtual ~HierBvar() = default;
	Rcpp::List returnMinnRes(int num_burn);
private:
	std::unique_ptr<HierMinn> _mn;
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

#endif