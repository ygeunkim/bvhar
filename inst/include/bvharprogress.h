#ifndef BVHARPROGRESS_H
#define BVHARPROGRESS_H

#include "bvharomp.h"
#include <Rcpp.h>

namespace bvhar {

class bvharprogress {
public:
	bvharprogress(int total, bool verbose) : _current(0), _total(total), _width(50), _verbose(verbose) {}
	virtual ~bvharprogress() = default;
	void increment() {
		if (omp_get_thread_num() == 0) {
			_current++;
		} else {
			_current.fetch_add(1, std::memory_order_relaxed);
		}
	}
	void update() {
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
private:
	std::atomic<int> _current;
	int _total;
	int _width;
	bool _verbose;
};

} // namespace bvhar

#endif // BVHARPROGRESS_H