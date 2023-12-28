#ifndef BVHARPROGRESS_H
#define BVHARPROGRESS_H

#include <Rcpp.h>
#include <atomic>

class bvharprogress {
private:
	std::atomic<int> _current;
	int _total;
	int _width;
	bool _verbose;
public:
	bvharprogress(int total, bool verbose);
	virtual ~bvharprogress() = default;
	void update();
	void increment();
	bool is_interrupted() const;
	void set_interrupt(bool flag);
};

#endif