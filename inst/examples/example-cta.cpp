#include <bvhar/ols>
#include <bvhar/triangular>
#include <cstdlib>

inline std::ostream& operator<<(std::ostream& os, const BvharList& dict) {
  os << "BVHAR_LIST (size: " << dict.size() << ") {\n";
  for (const auto& kv : dict) {
    os << "  $ " << kv.first << "\t: ";
    const std::any& val = kv.second;
    
    if (!val.has_value()) {
      os << "NULL";
    } else if (val.type() == typeid(int)) {
      os << std::any_cast<int>(val);
    } else if (val.type() == typeid(double)) {
      os << std::any_cast<double>(val);
    } else if (val.type() == typeid(bool)) {
      os << (std::any_cast<bool>(val) ? "TRUE" : "FALSE");
    } else if (val.type() == typeid(std::string)) {
      os << "\"" << std::any_cast<std::string>(val) << "\"";
    } else if (val.type() == typeid(Eigen::MatrixXd)) {
      auto mat = std::any_cast<Eigen::MatrixXd>(val);
      os << "<Eigen::MatrixXd " << mat.rows() << "x" << mat.cols() << ">";
    } else if (val.type() == typeid(Eigen::VectorXd)) {
      auto vec = std::any_cast<Eigen::VectorXd>(val);
      os << "<Eigen::VectorXd length " << vec.size() << ">";
    } else if (val.type() == typeid(Eigen::MatrixXi)) {
      auto mat = std::any_cast<Eigen::MatrixXi>(val);
      os << "<Eigen::MatrixXi " << mat.rows() << "x" << mat.cols() << ">";
    } else if (val.type() == typeid(Eigen::VectorXi)) {
      auto vec = std::any_cast<Eigen::VectorXi>(val);
      os << "<Eigen::VectorXi length " << vec.size() << ">";
    } else if (val.type() == typeid(std::vector<std::any>)) {
      os << "<BVHAR_PY_LIST length " << std::any_cast<std::vector<std::any>>(val).size() << ">";
    } else if (val.type() == typeid(std::vector<BvharList>)) {
      os << "<BVHAR_LIST_OF_LIST length " << std::any_cast<std::vector<BvharList>>(val).size() << ">";
    } else {
      os << "<Unknown C++ Type: " << val.type().name() << ">";
    }
    os << "\n";
  }
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<BvharList>& chains) {
  os << "BVHAR_LIST_OF_LIST (Total Chains: " << chains.size() << ") [\n";
  for (size_t i = 0; i < chains.size(); ++i) {
    os << "  === CHAIN " << (i + 1) << " ===\n";
    os << chains[i] << "\n";
  }
  os << "]";
  return os;
}

int main() {
#ifdef _OPENMP
  std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
	std::cout << "OpenMP not available in this machine." << std::endl;
#endif
	int num_chains = 3;
	int nthreads = 3;
	int num_iter = 500;
	int num_burn = 300;
	int thin = 2;
	int dim = 5;
	int num_data = 50;
	int lag = 3;
	bool include_mean = true;
	int dim_design = include_mean ? dim * lag + 1 : dim * lag;
	int num_coef = dim * dim_design;
	int num_alpha = include_mean ? num_coef - dim : num_coef;
	int num_eta = dim * (dim - 1) / 2;
	int seed = 1;
	BVHAR_BHRNG dgp_rng(seed);
	Eigen::MatrixXd var_coef(dim_design, dim);
	for (int i = 0; i < dim_design * dim; ++i) {
		var_coef(i) = baecon::bvhar::normal_rand(dgp_rng) / 5;
	}
	Eigen::MatrixXd var_cov = Eigen::MatrixXd::Identity(dim, dim) / 3;
	auto dgp_run = std::make_unique<baecon::bvhar::OlsSimulator>(
		num_data, 0, lag, Eigen::MatrixXd::Zero(lag, dim),
		var_coef, var_cov, 2, 1
	);
	Eigen::MatrixXd time_series = dgp_run->returnDgp();
	std::cout << "VAR(p=" << lag << ")\n"
		<< "Coefficient:\n" << var_coef << "\n"
		<< "Covariance:\n" << var_cov << "\n"
		<< "Data:\n" << time_series << std::endl;
	std::cout << "MCMC configuration\n"
		<< "Chains: " << num_chains << "\n"
		<< "Iteration: " << num_iter << "\n"
		<< "Burn-in: " << num_burn << "\n"
		<< "Thinning: " << thin << std::endl;
  try {
		Eigen::MatrixXd x = baecon::bvhar::build_x0(time_series, lag, include_mean);
		Eigen::MatrixXd y = baecon::bvhar::build_y0(time_series, lag, lag + 1);
		Eigen::MatrixXi grp_mat = Eigen::MatrixXi::Zero(dim * lag, dim);
		for (int i = 0; i < lag; ++i) {
			grp_mat.middleRows(i * dim, dim).setIdentity();
			grp_mat.middleRows(i * dim, dim).array() += 2 * i + 1;
		}
		std::set<int> unique_grp(grp_mat.data(), grp_mat.data() + grp_mat.size());
		int num_grp = unique_grp.size();
		Eigen::VectorXi grp_id(num_grp);
		int unique_id = 0;
		for (int id : unique_grp) {
			grp_id[unique_id++] = id;
		}
		Eigen::VectorXi own_id = Eigen::VectorXi::LinSpaced(lag, 2, 2 * lag);
		Eigen::VectorXi cross_id = Eigen::VectorXi::LinSpaced(lag, 1, 2 * lag);
		std::cout << "Group:\n" << grp_mat << "\n"
			<< "Group id: " << grp_id.transpose() << "\n"
			<< "Own id: " << own_id.transpose() << "\n"
			<< "Cross id: " << cross_id.transpose() << std::endl;
		BVHAR_LIST param_reg = BVHAR_CREATE_LIST(
			BVHAR_NAMED("shape") = Eigen::VectorXd::Constant(dim, 3.0),
			BVHAR_NAMED("scale") = Eigen::VectorXd::Constant(dim, 0.01)
		);
		int prior_type = 3;
		int contem_prior_type = 3;
		BVHAR_LIST param_prior{};
		BVHAR_LIST contem_prior{};
		BVHAR_LIST param_intercept = BVHAR_CREATE_LIST(
			BVHAR_NAMED("mean_non") = Eigen::VectorXd::Zero(dim),
			BVHAR_NAMED("sd_non") = .1
		);
		// BVHAR_LIST_OF_LIST param_init(num_chains);
		// BVHAR_LIST_OF_LIST contem_init(num_chains);
		// srand(1);
		// for (int i = 0; i < num_chains; ++i) {
		// 	param_init[i] = BVHAR_CREATE_LIST(
		// 		BVHAR_NAMED("init_coef") = Eigen::MatrixXd::Random(dim_design, dim),
		// 		BVHAR_NAMED("init_contem") = Eigen::VectorXd::Random(num_eta),
		// 		BVHAR_NAMED("init_diag") = Eigen::VectorXd::Random(dim).array().exp().matrix(),
		// 		BVHAR_NAMED("local_sparsity") = Eigen::VectorXd::Random(num_alpha).array().exp().matrix(),
		// 		BVHAR_NAMED("global_sparsity") = 1.0,
		// 		BVHAR_NAMED("group_sparsity") = Eigen::VectorXd::Random(num_grp).array().exp().matrix()
		// 	);
		// 	contem_init[i] = BVHAR_CREATE_LIST(
		// 		BVHAR_NAMED("local_sparsity") = Eigen::VectorXd::Random(num_eta).array().exp().matrix(),
		// 		BVHAR_NAMED("global_sparsity") = 1.0,
		// 		BVHAR_NAMED("group_sparsity") = Eigen::VectorXd::Random(1).array().exp().matrix()
		// 	);
		// }
		Eigen::VectorXi seed_chain = Eigen::VectorXi::Random(num_chains);
		std::cout << "Initialzing MCMC..." << std::endl;
		auto mcmc_run = std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
			num_chains, num_iter, num_burn, thin, x, y,
			param_reg, param_prior, param_intercept, prior_type,
			contem_prior, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain, true, nthreads
		);
		std::cout << "Running MCMC..." << std::endl;
		// BVHAR_LIST result = BVHAR_CAST_LIST(mcmc_run->returnRecords());
		BVHAR_LIST_OF_LIST result = mcmc_run->returnRecords();
		std::cout << "MCMC result:\n" << result << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Caught an error: " << e.what() << "\n";
    return 1;
  }
	return 0;
}
