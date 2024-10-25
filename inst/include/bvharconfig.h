#ifndef BVHARCONFIG_H
#define BVHARCONFIG_H

#include "bvhardraw.h"
#include <utility>

namespace bvhar {

struct RegParams;
struct RegInits;
struct RegRecords;
struct SparseRecords;
struct SsvsRecords;
struct GlobalLocalRecords;
struct HorseshoeRecords;
struct NgRecords;

struct RegParams {
	int _iter;
	Eigen::MatrixXd _x, _y;
	Eigen::VectorXd _sig_shp, _sig_scl, _mean_non;
	double _sd_non;
	bool _mean;
	int _dim, _dim_design, _num_design, _num_lowerchol, _num_coef, _num_alpha;

	RegParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec, LIST& intercept,
		bool include_mean
	)
	: _iter(num_iter), _x(x), _y(y),
		_sig_shp(CAST<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(CAST<Eigen::VectorXd>(spec["scale"])),
		_mean_non(CAST<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(CAST_DOUBLE(intercept["sd_non"])), _mean(include_mean),
		_dim(y.cols()), _dim_design(x.cols()), _num_design(y.rows()),
		_num_lowerchol(_dim * (_dim - 1) / 2), _num_coef(_dim * _dim_design),
		_num_alpha(_mean ? _num_coef - _dim : _num_coef) {}
};

struct RegInits {
	Eigen::MatrixXd _coef;
	Eigen::VectorXd _contem;

	RegInits(const RegParams& params) {
		_coef = (params._x.transpose() * params._x).llt().solve(params._x.transpose() * params._y); // OLS
		int dim = params._y.cols();
		int num_lowerchol = dim * (dim - 1) / 2;
		_contem = .001 * Eigen::VectorXd::Zero(num_lowerchol);
	}

	RegInits(LIST& init)
	: _coef(CAST<Eigen::MatrixXd>(init["init_coef"])),
		_contem(CAST<Eigen::VectorXd>(init["init_contem"])) {}
	
	// void updateCoef()

	// void updateImpact()

	// virtual void updateState()
};

struct RegRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)

	RegRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	RegRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}

	Eigen::VectorXd computeActivity(double level) {
		Eigen::VectorXd lower_ci(coef_record.cols());
		Eigen::VectorXd upper_ci(coef_record.cols());
		Eigen::VectorXd selection(coef_record.cols());
		for (int i = 0; i < coef_record.cols(); ++i) {
			// lower_ci[i] = quantile_lower(coef_record.col(i), level / 2);
			// upper_ci[i] = quantile_upper(coef_record.col(i), 1 - level / 2);
			selection[i] = quantile_lower(coef_record.col(i), level / 2) * quantile_upper(coef_record.col(i), 1 - level / 2) < 0 ? 0.0 : 1.1;
		}
		return selection;
	}
};

struct SparseRecords {
	Eigen::MatrixXd coef_record;
	Eigen::MatrixXd contem_coef_record;

	SparseRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	SparseRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}
	
	void assignRecords(int id, const Eigen::MatrixXd& coef_mat, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id) = coef_mat.reshaped();
		contem_coef_record.row(id) = contem_coef;
	}
};

struct SsvsRecords {
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd contem_dummy_record;
	Eigen::MatrixXd contem_weight_record;

	SsvsRecords() : coef_dummy_record(), coef_weight_record(), contem_dummy_record(), contem_weight_record() {}
	
	SsvsRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: coef_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_alpha)),
		coef_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		contem_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_lowerchol)),
		contem_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	SsvsRecords(
		const Eigen::MatrixXd& coef_dummy_record, const Eigen::MatrixXd& coef_weight_record,
		const Eigen::MatrixXd& contem_dummy_record, const Eigen::MatrixXd& contem_weight_record
	)
	: coef_dummy_record(coef_dummy_record), coef_weight_record(coef_weight_record),
		contem_dummy_record(contem_dummy_record), contem_weight_record(contem_weight_record) {}
	
	void assignRecords(int id, const Eigen::VectorXd& coef_dummy, const Eigen::VectorXd& coef_weight, const Eigen::VectorXd& contem_dummy, const Eigen::VectorXd& contem_weight) {
		coef_dummy_record.row(id) = coef_dummy;
		coef_weight_record.row(id) = coef_weight;
		contem_dummy_record.row(id) = contem_dummy;
		contem_weight_record.row(id) = contem_weight;
	}

	SsvsRecords returnRecords(int num_iter, int num_burn, int thin) const {
		SsvsRecords res_record(
			thin_record(coef_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(coef_weight_record, num_iter, num_burn, thin).derived(),
			thin_record(contem_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(contem_weight_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

struct GlobalLocalRecords {
	Eigen::MatrixXd local_record;
	Eigen::VectorXd global_record;

	GlobalLocalRecords() : local_record(), global_record() {}
	
	GlobalLocalRecords(int num_iter, int num_alpha)
	: local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)) {}
	
	GlobalLocalRecords(const Eigen::MatrixXd& local_record, const Eigen::VectorXd& global_record)
	: local_record(local_record), global_record(global_record) {}
	
	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const double global_lev) {
		local_record.row(id) = local_lev;
		global_record[id] = global_lev;
	}

	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	virtual void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	GlobalLocalRecords returnGlRecords(int num_iter, int num_burn, int thin) const {
		GlobalLocalRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

struct HorseshoeRecords : public GlobalLocalRecords {
	Eigen::MatrixXd group_record;
	Eigen::MatrixXd shrink_record;

	HorseshoeRecords() : GlobalLocalRecords(), group_record(), shrink_record() {}
	
	HorseshoeRecords(int num_iter, int num_alpha, int num_grp)
	: GlobalLocalRecords(num_iter, num_alpha),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)) {}
	
	HorseshoeRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& group_record, const Eigen::VectorXd& global_record, const Eigen::MatrixXd& shrink_record)
	: GlobalLocalRecords(local_record, global_record),
		group_record(group_record), shrink_record(shrink_record) {}
	
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {
		shrink_record.row(id) = shrink_fac;
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {}

	HorseshoeRecords returnHsRecords(int num_iter, int num_burn, int thin) const {
		HorseshoeRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(group_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived(),
			thin_record(shrink_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

struct NgRecords : public GlobalLocalRecords {
	Eigen::MatrixXd group_record;

	NgRecords() : GlobalLocalRecords(), group_record() {}
	
	NgRecords(int num_iter, int num_alpha, int num_grp)
	: GlobalLocalRecords(num_iter, num_alpha),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)) {}
	
	NgRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& group_record, const Eigen::VectorXd& global_record)
	: GlobalLocalRecords(local_record, global_record), group_record(group_record) {}
	
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {}
	
	void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	NgRecords returnNgRecords(int num_iter, int num_burn, int thin) const {
		NgRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(group_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

} // namespace bvhar

#endif // BVHARCONFIG_H