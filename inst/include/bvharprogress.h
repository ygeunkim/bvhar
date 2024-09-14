#ifndef BVHARPROGRESS_H
#define BVHARPROGRESS_H

#include "bvharomp.h"
#ifdef USE_RCPP
	#include <Rcpp.h>
	#define COUT Rcpp::Rcout
	#define ENDL "\n"
	#define FLUSH Rcpp::Rcout.flush()
#else
	#include <iostream>
	#define COUT std::cout
	#define ENDL std::endl
	#define FLUSH std::cout.flush()
#endif

namespace bvhar {

class bvharprogress;

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
		// Rcpp::Rcout << "\r";
		COUT << "\r";
		for (int i = 0; i < _width; i++) {
			if (i < (percent * _width / 100)) {
				// Rcpp::Rcout << "#";
				std::cout << "#";
			} else {
				// Rcpp::Rcout << " ";
				COUT << " ";
			}
		}
		// Rcpp::Rcout << " " << percent << "%";
		// Rcpp::Rcout.flush();
		COUT << " " << percent << "%";
		FLUSH;
		if (_current >= _total) {
			// Rcpp::Rcout << "\n";
			COUT << ENDL;
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