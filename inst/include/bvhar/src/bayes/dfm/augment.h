/**
 * @file augment.h
 * @brief Header including factor augmenters
 */
#ifndef BVHAR_BAYES_DFM_AUGMENT_H
#define BVHAR_BAYES_DFM_AUGMENT_H

#include "./config.h"

namespace bvhar {

class FactorAugmenter;
class FactorNormalAugmenter;

class FactorAugmenter {
public:
	FactorAugmenter(int num_iter, int num_design, const DfmParams& params)
	: num_iter(num_iter), num_design(num_design), dim(params._dim),
		size_factor(params._size_factor), lag(params._lag),
		resid(Eigen::MatrixXd::Zero(num_design, dim)),
		factor_design(Eigen::MatrixXd::Zero(size_factor, num_design)) {}
	virtual ~FactorAugmenter() = default;

	void appendDesign(Eigen::Ref<Eigen::MatrixXd> x) {
		// x.bottomRightCorner(num_design, size_factor) = factor_design.transpose(); // diag(X_0, F_0)
		x.rightCols(size_factor) = factor_design.transpose(); // (X_0, F_0)
	}

	void updateResid(
		Eigen::Ref<const Eigen::MatrixXd> x, Eigen::Ref<const Eigen::MatrixXd> y,
		Eigen::Ref<const Eigen::MatrixXd> coef_mat
	) {
		resid = y - x.leftCols(x.cols() - size_factor) * coef_mat.topRows(coef_mat.rows() - size_factor);
	}
	
	// virtual void updateFactor(
	// 	Eigen::Ref<const Eigen::MatrixXd> coef_mat,
	// 	Eigen::Ref<const Eigen::MatrixXd> contem_mat, Eigen::Ref<const Eigen::VectorXd> diag_vec,
	// 	BVHAR_BHRNG& rng
	// ) {}

	virtual void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> coef_mat,
		Eigen::Ref<const Eigen::MatrixXd> contem_mat, Eigen::Ref<const Eigen::MatrixXd> diag_vec_t,
		BVHAR_BHRNG& rng
	) {}

	void appendRecords(BVHAR_LIST& list) {
		dfm_record->appendRecords(list);
	}

	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin) const {
		return dfm_record->returnRecords<RecordType>(num_iter, num_burn, thin);
	}

	virtual void updateRecords(int id) {}

protected:
	int num_iter, num_design, dim, size_factor, lag;
	Eigen::MatrixXd resid; // Y0 - X0 A vs Y_j - X_j A(j)
	Eigen::MatrixXd factor_design; // Transpose of (f_{p + 1}, ..., f_T)^T
	// Eigen::VectorXd factor_t; // f_t
	std::unique_ptr<DfmRecords> dfm_record;
};

class FactorNormalAugmenter : public FactorAugmenter {
public:
	FactorNormalAugmenter(int num_iter, int num_design, const DfmParams& params)
	: FactorAugmenter(num_iter, num_design, params) {
		dfm_record = std::make_unique<DfmRecords>(num_iter, num_design, size_factor);
	}
	virtual ~FactorNormalAugmenter() = default;
	
	// void updateFactor(
	// 	Eigen::Ref<const Eigen::MatrixXd> coef_mat,
	// 	Eigen::Ref<const Eigen::MatrixXd> contem_mat, Eigen::Ref<const Eigen::VectorXd> diag_vec,
	// 	BVHAR_BHRNG& rng
	// ) override {
	// 	Eigen::MatrixXd sig_lower = contem_mat.triangularView<Eigen::UnitLower>().solve(diag_vec.asDiagonal().toDenseMatrix());
	// 	for (int i = 0; i < num_design; ++i) {
	// 		draw_normal_factor(factor_design.col(i), resid.row(i), coef_mat.bottomRows(size_factor).transpose(), sig_lower, rng);
	// 	}
	// }

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> coef_mat,
		Eigen::Ref<const Eigen::MatrixXd> contem_mat, Eigen::Ref<const Eigen::MatrixXd> diag_vec_t,
		BVHAR_BHRNG& rng
	) override {
		for (int i = 0; i < num_design; ++i) {
			Eigen::MatrixXd sig_lower = contem_mat.triangularView<Eigen::UnitLower>().solve(diag_vec_t.row(i).asDiagonal().toDenseMatrix());
			draw_normal_factor(factor_design.col(i), resid.row(i), coef_mat.bottomRows(size_factor).transpose(), sig_lower, rng);
		}
	}

	void updateRecords(int id) override {
		dfm_record->assignRecords(id, factor_design, num_design, size_factor);
	}
};

} // namespace bvhar

#endif // BVHAR_BAYES_DFM_AUGMENT_H