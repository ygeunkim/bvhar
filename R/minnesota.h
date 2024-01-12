#ifndef MINNESOTA_H
#define MINNESOTA_H

#include <RcppEigen.h>

class Minnesota {
public:
	Minnesota(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
	virtual ~Minnesota() = default;

};


#endif