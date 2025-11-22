#ifndef BVHAR_OLS_OLS_H
#define BVHAR_OLS_OLS_H

#include "../math/design.h"
#include <memory> // std::unique_ptr in source file

namespace baecon {
namespace bvhar {

struct OlsFit;
struct StructuralFit;
class MultiOls;
class LltOls;
class QrOls;
class OlsVar;
class OlsVhar;

struct OlsFit {
	Eigen::MatrixXd _coef;
	int _ord; // p of VAR or month of VHAR

	OlsFit(const Eigen::MatrixXd& coef_mat, int ord) : _coef(coef_mat), _ord(ord) {}
};

struct StructuralFit : public OlsFit {
	int _lag_max;
	int dim;
	// int ma_rows;
	Eigen::MatrixXd _vma; // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
	Eigen::MatrixXd _cov;

	StructuralFit(const Eigen::MatrixXd& coef_mat, int ord, const Eigen::MatrixXd& cov_mat)
	: OlsFit(coef_mat, ord), dim(coef_mat.cols()), _cov(cov_mat) {}
	
	// StructuralFit(const Eigen::MatrixXd& coef_mat, int ord, int lag_max, const Eigen::MatrixXd& cov_mat)
	// : OlsFit(coef_mat, ord), _lag_max(lag_max),
	// 	dim(coef_mat.cols()), ma_rows(dim * (_lag_max + 1)),
	// 	_vma(Eigen::MatrixXd::Zero(ma_rows, dim)), _cov(cov_mat) {
	// 	int num_full_rows = _lag_max < _ord ? dim * _ord : ma_rows;
	// 	Eigen::MatrixXd full_coef = Eigen::MatrixXd::Zero(num_full_rows, dim); // same size with VMA coefficient matrix
	// 	full_coef.topRows(dim * _ord) = _coef.topRows(dim * _ord); // fill first mp row with VAR coefficient matrix
	// 	_vma.topRows(dim) = Eigen::MatrixXd::Identity(dim, dim); // W0 = I_k
	// 	for (int i = 1; i < (_lag_max + 1); ++i) {
	// 		for (int j = 0; j < i; ++j) {
	// 			_vma.middleRows(i * dim, dim) += full_coef.middleRows(j * dim, dim) * _vma.middleRows((i - j - 1) * dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
	// 		}
	// 	}
	// }
};

class MultiOls {
public:
	MultiOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	: design(x), response(y),
		dim(response.cols()), num_design(response.rows()), dim_design(design.cols()) {
		coef = Eigen::MatrixXd::Zero(dim_design, dim);
		yhat = Eigen::MatrixXd::Zero(num_design, dim);
		resid = Eigen::MatrixXd::Zero(num_design, dim);
		cov = Eigen::MatrixXd::Zero(dim, dim);
	}
	virtual ~MultiOls() = default;
	virtual void estimateCoef() {
		coef = (design.transpose() * design).inverse() * design.transpose() * response; // return coef -> use in OlsVar
	}
	virtual void fitObs() {
		yhat = design * coef;
		resid = response - yhat;
	}
	void estimateCov() {
		cov = resid.transpose() * resid / (num_design - dim_design);
	}
	BVHAR_LIST returnOlsRes() {
		estimateCoef();
		fitObs();
		estimateCov();
		return BVHAR_CREATE_LIST(
			BVHAR_NAMED("coefficients") = coef,
			BVHAR_NAMED("fitted.values") = yhat,
			BVHAR_NAMED("residuals") = resid,
			BVHAR_NAMED("covmat") = cov,
			BVHAR_NAMED("df") = dim_design,
			BVHAR_NAMED("m") = dim,
			BVHAR_NAMED("obs") = num_design,
			BVHAR_NAMED("y0") = response
		);
	}
	Eigen::MatrixXd returnCoef() {
		estimateCoef();
		fitObs();
		estimateCov();
		return coef;
	}
	OlsFit returnOlsFit(int ord) {
		estimateCoef();
		fitObs();
		estimateCov();
		OlsFit res(coef, ord);
		return res;
	}
	StructuralFit returnStructuralFit(int ord) {
		estimateCoef();
		fitObs();
		estimateCov();
		// StructuralFit res(coef, ord, lag_max, cov);
		StructuralFit res(coef, ord, cov);
		return res;
	}
	StructuralFit returnStructuralFit(const Eigen::MatrixXd& trans_mat, int ord) {
		estimateCoef();
		fitObs();
		estimateCov();
		// StructuralFit res(trans_mat.transpose() * coef, ord, lag_max, cov);
		StructuralFit res(trans_mat.transpose() * coef, ord, cov);
		return res;
	}
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
	LltOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : MultiOls(x, y) {
		llt_selfadjoint.compute(design.transpose() * design);
	}
	virtual ~LltOls() = default;
	void estimateCoef() override {
		coef = llt_selfadjoint.solve(design.transpose() * response);
	}
private:
	Eigen::LLT<Eigen::MatrixXd> llt_selfadjoint;
};

class QrOls : public MultiOls {
public:
	QrOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : MultiOls(x, y) {
		qr_design.compute(design);
	}
	virtual ~QrOls() = default;
	void estimateCoef() override {
		coef = qr_design.solve(response);
	}
private:
	Eigen::HouseholderQR<Eigen::MatrixXd> qr_design;
};

inline std::unique_ptr<MultiOls> initialize_ols(const Eigen::MatrixXd& design, const Eigen::MatrixXd& response, int method) {
	std::unique_ptr<MultiOls> ols_ptr;
	switch (method) {
		case 1: {
			ols_ptr = std::make_unique<MultiOls>(design, response);
			return ols_ptr;
		}
		case 2: {
			ols_ptr = std::make_unique<LltOls>(design, response);
			return ols_ptr;
		}
		case 3: {
			ols_ptr = std::make_unique<QrOls>(design, response);
			return ols_ptr;
		}
	}
	return ols_ptr;
}

class OlsInterface {
public:
	OlsInterface() {}
	virtual ~OlsInterface() = default;
	virtual BVHAR_LIST returnOlsRes() = 0;
	virtual Eigen::MatrixXd returnCoef() = 0;
	virtual OlsFit returnOlsFit() = 0;
	virtual StructuralFit returnStructuralFit() = 0;
};

class OlsVar : public OlsInterface {
public:
	OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean, int method)
	: lag(lag), const_term(include_mean), data(y) {
		response = build_y0(data, lag, lag + 1);
		design = build_x0(data, lag, const_term);
		_ols = initialize_ols(design, response, method);
	}
	OlsVar(const Eigen::MatrixXd& y, const Eigen::MatrixXd& exogen, int lag, int exogen_lag, const bool include_mean, int method)
	: lag(lag), const_term(include_mean), data(y) {
		response = build_y0(data, lag, lag + 1);
		design = build_x0(data, exogen, lag, exogen_lag, const_term);
		_ols = initialize_ols(design, response, method);
	}
	virtual ~OlsVar() = default;
	BVHAR_LIST returnOlsRes() override {
		BVHAR_LIST ols_res = _ols->returnOlsRes();
		ols_res["p"] = lag;
		ols_res["totobs"] = data.rows();
		ols_res["process"] = "VAR";
		ols_res["type"] = const_term ? "const" : "none";
		ols_res["design"] = design;
		ols_res["y"] = data;
		return ols_res;
	}
	Eigen::MatrixXd returnCoef() override {
		return _ols->returnCoef();
	}
	OlsFit returnOlsFit() override{
		OlsFit res = _ols->returnOlsFit(lag);
		return res;
	}
	StructuralFit returnStructuralFit() override {
		StructuralFit res = _ols->returnStructuralFit(lag);
		return res;
	}
protected:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	std::unique_ptr<MultiOls> _ols;
	Eigen::MatrixXd response;
	Eigen::MatrixXd design;
};

class OlsVhar : public OlsInterface {
public:
	OlsVhar(const Eigen::MatrixXd& y, int week, int month, const bool include_mean, int method)
	: week(week), month(month), const_term(include_mean), data(y) {
		response = build_y0(data, month, month + 1);
		har_trans = build_vhar(response.cols(), week, month, const_term);
		var_design = build_x0(data, month, const_term);
		design = var_design * har_trans.transpose();
		_ols = initialize_ols(design, response, method);
	}
	OlsVhar(const Eigen::MatrixXd& y, const Eigen::MatrixXd& exogen, int week, int month, int exogen_lag, const bool include_mean, int method)
	: week(week), month(month), const_term(include_mean), data(y),
		response(build_y0(data, month, month + 1)) {
		// har_trans = build_vhar(response.cols(), exogen.cols(), week, month, const_term);
		har_trans = build_vhar(response.cols(), week, month, const_term);
		// var_design = build_x0(data, exogen, month, month, const_term);
		// design = var_design * har_trans.transpose();
		var_design = build_x0(data, exogen, month, exogen_lag, const_term);
		int dim_design = include_mean ? month * response.cols() + 1 : month * response.cols();
		int dim_har = include_mean ? 3 * response.cols() + 1 : 3 * response.cols();
		int dim_exogen = (exogen_lag + 1) * exogen.cols();
		design = Eigen::MatrixXd::Zero(data.rows() - month, dim_har + (exogen_lag + 1) * exogen.cols());
		design.leftCols(dim_har) = var_design.leftCols(dim_design) * har_trans.transpose();
		design.rightCols(dim_exogen) = var_design.rightCols(dim_exogen);
		_ols = initialize_ols(design, response, method);
	}
	virtual ~OlsVhar() = default;
	BVHAR_LIST returnOlsRes() override {
		BVHAR_LIST ols_res = _ols->returnOlsRes();
		ols_res["p"] = 3;
		ols_res["week"] = week;
		ols_res["month"] = month;
		ols_res["totobs"] = data.rows();
		ols_res["process"] = "VHAR";
		ols_res["type"] = const_term ? "const" : "none";
		ols_res["HARtrans"] = har_trans;
		ols_res["design"] = var_design;
		ols_res["y"] = data;
		return ols_res;
	}
	Eigen::MatrixXd returnCoef() override {
		return _ols->returnCoef();
	}
	OlsFit returnOlsFit() override {
		OlsFit res = _ols->returnOlsFit(month);
		res._ord = month;
		return res;
	}
	StructuralFit returnStructuralFit() override {
		// StructuralFit res = _ols->returnStructuralFit(har_trans, month);
		StructuralFit res = _ols->returnStructuralFit(month);
		return res;
	}
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

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_OLS_OLS_H