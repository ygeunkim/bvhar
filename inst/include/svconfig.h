#ifndef SVCONFIG_H
#define SVCONFIG_H

#include "bvharconfig.h"

namespace bvhar {

// MCMC records
struct SvRecords;

struct SvRecords : public RegRecords {
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
	
	SvRecords() : RegRecords(), lvol_sig_record(), lvol_init_record(), lvol_record() {}

	SvRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		lvol_sig_record(Eigen::MatrixXd::Ones(num_iter + 1, dim)),
		lvol_init_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		lvol_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim)) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(alpha_record, a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols()), a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {
		coef_record << alpha_record, c_record;
	}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) override {}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) override {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		lvol_record.row(id) = lvol_draw.transpose().reshaped();
		lvol_sig_record.row(id) = lvol_sig;
		lvol_init_record.row(id) = lvol_init;
	}

	void appendRecords(LIST& list) override {
		list["h_record"] = lvol_record;
		list["h0_record"] = lvol_init_record;
		list["sigh_record"] = lvol_sig_record;
	}

	void subsetStable(int num_alpha, double threshold) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		Eigen::MatrixXd var_record = coef_record.leftCols(num_alpha) * kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), har_trans);
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(var_record.row(i).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(har_trans.transpose() * coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}
};

} // namespace bvhar

#endif // SVCONFIG_H