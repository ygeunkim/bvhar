#ifndef BVHAR_CORE_PROGRESS_H
#define BVHAR_CORE_PROGRESS_H

#include "commondefs.h"
#include "omp.h"

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
		BVHAR_COUT << "\r";
		for (int i = 0; i < _width; i++) {
			if (i < (percent * _width / 100)) {
				BVHAR_COUT << "#";
			} else {
				BVHAR_COUT << " ";
			}
		}
		BVHAR_COUT << " " << percent << "%";
		BVHAR_FLUSH;
		if (_current >= _total) {
			BVHAR_COUT << BVHAR_ENDL;
		}
	}
private:
	std::atomic<int> _current;
	int _total;
	int _width;
	bool _verbose;
};

} // namespace bvhar

#endif // BVHAR_CORE_PROGRESS_H