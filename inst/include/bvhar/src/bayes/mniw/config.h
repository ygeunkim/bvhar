#ifndef BVHAR_BAYES_MNIW_CONFIG_H
#define BVHAR_BAYES_MNIW_CONFIG_H

#include "../misc/draw.h"
#include "../../math/design.h"
// #include <memory> // std::unique_ptr
#include "../../core/progress.h"

namespace bvhar {

struct MinnSpec;
struct BvarSpec;
struct BvharSpec;
struct MinnFit;
struct MhMinnInits;
struct MhMinnSpec;
struct MinnRecords;
struct MhMinnRecords;

struct MinnSpec {
	Eigen::VectorXd _sigma;
	double _lambda;
	double _eps;

	MinnSpec(BVHAR_LIST& bayes_spec)
	: _sigma(BVHAR_CAST<Eigen::VectorXd>(bayes_spec["sigma"])),
		_lambda(BVHAR_CAST_DOUBLE(bayes_spec["lambda"])),
		_eps(BVHAR_CAST_DOUBLE(bayes_spec["eps"])) {}
};

struct BvarSpec : public MinnSpec {
	Eigen::VectorXd _delta;

	BvarSpec(BVHAR_LIST& bayes_spec)
	: MinnSpec(bayes_spec),
		_delta(BVHAR_CAST<Eigen::VectorXd>(bayes_spec["delta"])) {}
};

struct BvharSpec : public MinnSpec {
	Eigen::VectorXd _daily;
	Eigen::VectorXd _weekly;
	Eigen::VectorXd _monthly;

	BvharSpec(BVHAR_LIST& bayes_spec)
	: MinnSpec(bayes_spec),
		_daily(BVHAR_CAST<Eigen::VectorXd>(bayes_spec["daily"])),
		_weekly(BVHAR_CAST<Eigen::VectorXd>(bayes_spec["weekly"])),
		_monthly(BVHAR_CAST<Eigen::VectorXd>(bayes_spec["monthly"])) {}
};

struct MinnFit {
	Eigen::MatrixXd _coef;
	Eigen::MatrixXd _prec;
	Eigen::MatrixXd _iw_scale;
	double _iw_shape;
	
	MinnFit(const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& prec_mat, const Eigen::MatrixXd& iw_scale, double iw_shape)
	: _coef(coef_mat), _prec(prec_mat), _iw_scale(iw_scale), _iw_shape(iw_shape) {}
};

struct MhMinnInits {
	double _lambda;
	Eigen::VectorXd _psi;
	Eigen::MatrixXd _hess;
	double _acc_scale;
	// Eigen::VectorXd _delta;

	MhMinnInits(BVHAR_LIST& init) {
		Eigen::VectorXd par = BVHAR_CAST<Eigen::VectorXd>(init["par"]);
		_lambda = par[0];
		_psi = par.tail(par.size() - 1);
		_hess = BVHAR_CAST<Eigen::MatrixXd>(init["hessian"]);
		_acc_scale = BVHAR_CAST_DOUBLE(init["scale_variance"]);
		// _delta = BVHAR_CAST<Eigen::VectorXd>(spec["delta"]);
	}
};

struct MhMinnSpec {
	double _gam_shape;
	double _gam_rate;
	double _invgam_shape;
	double _invgam_scl;

	MhMinnSpec(BVHAR_LIST& lambda, BVHAR_LIST& psi) {
		// Rcpp::List lambda = spec["lambda"];
		// Rcpp::List psi = spec["sigma"];
		// Rcpp::List param = lambda["param"];
		Eigen::VectorXd lam_param = BVHAR_CAST<Eigen::VectorXd>(lambda["param"]);
		_gam_shape = lam_param[0];
		_gam_rate = lam_param[1];
		Eigen::VectorXd psi_param = BVHAR_CAST<Eigen::VectorXd>(psi["param"]);
		_invgam_shape = psi_param[0];
		_invgam_scl = psi_param[1];
	}
};

struct MinnRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd sig_record;

	MinnRecords(int num_iter, int dim, int dim_design)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim_design)),
		sig_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim)) {}
	
	MinnRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& sig_record)
	: coef_record(alpha_record), sig_record(sig_record) {}
	
	void assignRecords(int id, std::vector<Eigen::MatrixXd>& mniw_draw) {
		coef_record.row(id) = vectorize_eigen(mniw_draw[0]);
		sig_record.row(id) = vectorize_eigen(mniw_draw[1]);
	}
};

struct MhMinnRecords {
	// Eigen::MatrixXd coef_record; // alpha in VAR
	// Eigen::MatrixXd sig_record;
	Eigen::VectorXd lam_record;
	Eigen::MatrixXd psi_record;
	VectorXb accept_record;
	// int _dim;
	
	// MhMinnRecords(int num_iter, int dim, int dim_design)
	MhMinnRecords(int num_iter, int dim)
	// : coef_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim_design)),
	// 	sig_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim)),
	: lam_record(Eigen::VectorXd::Zero(num_iter + 1)),
		psi_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		accept_record(VectorXb(num_iter + 1)) {}
	void assignRecords(
		int id,
		// std::vector<Eigen::MatrixXd>& mniw_draw,
		double lambda, Eigen::Ref<Eigen::VectorXd> psi, bool is_accept
	) {
		// coef_record.row(id) = vectorize_eigen(mniw_draw[0]);
		// sig_record.row(id) = vectorize_eigen(mniw_draw[1]);
		lam_record[id] = lambda;
		psi_record.row(id) = psi;
		accept_record[id] = is_accept;
	}
};

} // namespace bvhar

#endif // BVHAR_BAYES_MNIW_CONFIG_H