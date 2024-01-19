#ifndef BVHARPROGRESS_H
#define BVHARPROGRESS_H

#include "bvharomp.h"
#include <Rcpp.h>
#include <atomic>

class bvharprogress {
public:
	bvharprogress(int total, bool verbose);
	virtual ~bvharprogress() = default;
	void increment();
	void update();
private:
	std::atomic<int> _current;
	int _total;
	int _width;
	bool _verbose;
};

#endif