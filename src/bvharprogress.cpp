#include "bvharprogress.h"

bvharprogress::bvharprogress(int total, bool verbose) : _current(0), _total(total), _width(50), _verbose(verbose) {}

void bvharprogress::increment() {
	std::lock_guard<std::mutex> lock(mtx);
	_current++;
}

std::string bvharprogress::progress() {
	std::lock_guard<std::mutex> lock(mtx);
	if (!_verbose) {
		return ""; // not display when verbose is false
	}
	int percent = _current * 100 / _total;
	oss << "\r";
	for (int i = 0; i < _width; i++) {
		if (i < (percent * _width / 100)) {
			oss << "#";
		} else {
			oss << " ";
		}
	}
	oss << " " << percent << "%";
	if (_current >= _total) {
		oss << "\n";
	}
	return oss.str();
}

void bvharprogress::update() {
	std::lock_guard<std::mutex> lock(mtx);
	if (!_verbose) {
		return; // not display when verbose is false
	}
	int percent = _current * 100 / _total;
	// int percent;
	// std::ostringstream oss;
	// Rcpp::Rcout << "\r";
	oss << "\r";
	for (int i = 0; i < _width; i++) {
		if (i < (percent * _width / 100)) {
			// Rcpp::Rcout << "#";
			oss << "#";
		} else {
			// Rcpp::Rcout << " ";
			oss << " ";
		}
	}
	// Rcpp::Rcout << " " << percent << "%";
	// Rcpp::Rcout.flush();
	oss << " " << percent << "%";
	if (_current >= _total) {
		// Rcpp::Rcout << "\n";
		oss << "\n";
	}
	Rcpp::Rcout << oss.str();
	Rcpp::Rcout.flush();
}
