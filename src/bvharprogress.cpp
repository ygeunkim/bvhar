#include "bvharprogress.h"

bvharprogress::bvharprogress(int total, bool verbose) : _total(total), _width(50), _current(0), _verbose(verbose) {}

void bvharprogress::increment() {
	_current++;
}

void bvharprogress::update() {
	if (!_verbose) {
		return; // not display when verbose is false
	}
	int percent = _current * 100 / _total;
	std::cout << "\r";
	for (int i = 0; i < _width; i++) {
		if (i < (percent * _width / 100)) {
			std::cout << "#";
		} else {
			std::cout << " ";
		}
	}
	std::cout << " " << percent << "%";
	std::flush(std::cout);
	if (_current >= _total) {
		std::cout << "\n";
	}
}
