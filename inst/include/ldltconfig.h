#ifndef LDLTCONFIG_H
#define LDLTCONFIG_H

#include "bvharconfig.h"

namespace bvhar {

// Initialization
struct LdltInits;
struct HierminnInits;
struct SsvsInits;
struct GlInits;
struct HsInits;
struct NgInits;
// MCMC records
struct LdltRecords;

struct LdltInits : public RegInits {
	Eigen::VectorXd _diag;

	LdltInits(LIST& init)
	: RegInits(init),
		_diag(CAST<Eigen::VectorXd>(init["init_diag"])) {}
	
	// void updateState() override
};

struct HierminnInits : public LdltInits {
	double _own_lambda;
	double _cross_lambda;
	double _contem_lambda;

	HierminnInits(LIST& init)
	: LdltInits(init),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
};

struct SsvsInits : public LdltInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight;
	Eigen::VectorXd _contem_weight;
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _contem_slab;

	SsvsInits(LIST& init)
	: LdltInits(init),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])) {}
};

struct GlInits : public LdltInits {
	Eigen::VectorXd _init_local;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	GlInits(LIST& init)
	: LdltInits(init),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

struct HsInits : public GlInits {
	Eigen::VectorXd _init_group;
	
	HsInits(LIST& init)
	: GlInits(init),
		_init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

struct NgInits : public HsInits {
	Eigen::VectorXd _init_local_shape;
	double _init_contem_shape;

	NgInits(LIST& init)
	: HsInits(init),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
};

struct LdltRecords : public RegRecords {
	Eigen::MatrixXd fac_record; // d_1, ..., d_m in D of LDLT

	LdltRecords() : RegRecords(), fac_record() {}

	LdltRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		fac_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)) {}
	
	LdltRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record)
	: RegRecords(alpha_record, a_record), fac_record(d_record) {}

	LdltRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols()), a_record),
		fac_record(d_record) {
		coef_record << alpha_record, c_record;
	}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) override {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		fac_record.row(id) = diag_vec.array();
	}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) override {}

	void appendRecords(LIST& list) override {
		list["d_record"] = fac_record;
	}

	void subsetStable(int num_alpha, double threshold) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		// Eigen::MatrixXd var_record = coef_record.leftCols(num_alpha) * kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), har_trans);
		for (int i = 0; i < coef_record.rows(); ++i) {
			// if (is_stable(var_record.row(i).reshaped(nrow_coef, dim))) {
			// 	stable_id.push_back(i);
			// }
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold, har_trans)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(har_trans.transpose() * coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}
};

} // namespace bvhar

#endif // LDLTCONFIG_H