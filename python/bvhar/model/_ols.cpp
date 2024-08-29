#include <pybind11/eigen.h>
#include <ols.h>

py::dict bvhar::MultiOls::returnOlsRes() {
	this->fit();
	// py::dict res;
	// res["coefficients"] = this->coef;
	// res["fitted.values"] = this->yhat;
	// res["residuals"] = this->resid;
	// res["covmat"] = this->cov;
	// res["df"] = this->dim_design;
	// res["m"] = this->dim;
	// res["obs"] = this->num_design;
	// res["y0"] = this->response;
	// return res;
	return py::dict(
		py::arg("coefficients") = this->coef,
		py::arg("fitted.values") = this->yhat,
		py::arg("residuals") = this->resid,
		py::arg("covmat") = this->cov,
		py::arg("df") = this->dim_design,
		py::arg("m") = this->dim,
		py::arg("obs") = this->num_design,
		py::arg("y0") = this->response
	);
}

py::dict bvhar::OlsVar::returnOlsRes() {
	py::dict ols_res = this->_ols->returnOlsRes();
	// py::dict ols_res;
	ols_res["p"] = this->lag;
	ols_res["totobs"] = this->data.rows();
	ols_res["process"] = "VAR";
	ols_res["type"] = this->const_term ? "const" : "none";
	ols_res["design"] = this->design;
	ols_res["y"] = this->data;
	return ols_res;
}

py::dict bvhar::OlsVhar::returnOlsRes() {
	py::dict ols_res = this->_ols->returnOlsRes();
	ols_res["p"] = 3;
	ols_res["week"] = this->week;
	ols_res["month"] = this->month;
	ols_res["totobs"] = this->data.rows();
	ols_res["process"] = "VHAR";
	ols_res["type"] = this->const_term ? "const" : "none";
	ols_res["HARtrans"] = this->har_trans;
	ols_res["design"] = this->var_design;
	ols_res["y"] = this->data;
	return ols_res;
}

PYBIND11_MODULE(_ols, m) {
	m.doc() = "OLS for VAR and VHAR";

  py::class_<bvhar::MultiOls>(m, "MultiOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
		.def("estimateCoef", &bvhar::MultiOls::estimateCoef)
		.def("fitObs", &bvhar::MultiOls::fitObs)
		.def("estimateCov", &bvhar::MultiOls::estimateCov)
		.def("fit", &bvhar::MultiOls::fit)
    .def("return_ols_res", &bvhar::MultiOls::returnOlsRes);
	
	py::class_<bvhar::LltOls, bvhar::MultiOls>(m, "LltOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<bvhar::QrOls, bvhar::MultiOls>(m, "QrOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<bvhar::OlsVar>(m, "OlsVar")
		.def(
			py::init<const Eigen::MatrixXd&, int, const bool, int>(),
			py::arg("y"), py::arg("lag"), py::arg("include_mean"), py::arg("method")
		)
		.def("return_ols_res", &bvhar::OlsVar::returnOlsRes);
	
	py::class_<bvhar::OlsVhar>(m, "OlsVhar")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, const bool, int>(),
			py::arg("y"), py::arg("week"), py::arg("month"), py::arg("include_mean"), py::arg("method")
		)
		.def("return_ols_res", &bvhar::OlsVhar::returnOlsRes);
}
