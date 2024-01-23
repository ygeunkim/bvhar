#include "bvharprogress.h"

namespace bvhar {

bvharprogress::bvharprogress(int total, bool verbose) : _current(0), _total(total), _width(50), _verbose(verbose) {}

void bvharprogress::increment() {
	if (omp_get_thread_num() == 0) {
		_current++;
	} else {
		_current.fetch_add(1, std::memory_order_relaxed);
	}
}

void bvharprogress::update() {
	if (!_verbose || omp_get_thread_num() != 0) {
		return; // not display when verbose is false
	}
	int percent = _current * 100 / _total;
	Rcpp::Rcout << "\r";
	for (int i = 0; i < _width; i++) {
		if (i < (percent * _width / 100)) {
			Rcpp::Rcout << "#";
		} else {
			Rcpp::Rcout << " ";
		}
	}
	Rcpp::Rcout << " " << percent << "%";
	Rcpp::Rcout.flush();
	if (_current >= _total) {
		Rcpp::Rcout << "\n";
	}
}

} // namespace bvhar