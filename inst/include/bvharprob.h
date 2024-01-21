#ifndef BVHARPROB_H
#define BVHARPROB_H

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/beta_distribution.hpp>

#define gammafn Rf_gammafn
#define lgammafn Rf_lgammafn
#define binom_rand Rf_rbinom
#define gamma_dens Rf_dgamma

inline double normal_rand(boost::random::mt19937& rng) {
	boost::random::normal_distribution<> rdist(0.0, 1.0);
	return rdist(rng);
}

inline double chisq_rand(double df) {
	return Rf_rchisq(df);
}

inline double chisq_rand(double df, unsigned int seed) {
	boost::random::mt19937 rng(seed);
	boost::random::chi_squared_distribution<> rdist(df);
	return rdist(rng);
}

inline double gamma_rand(double shp, double scl) {
	return Rf_rgamma(shp, scl); // 2nd: scale
}

inline double gamma_rand(double shp, double scl, boost::random::mt19937& rng) {
	boost::random::gamma_distribution<> rdist(shp, scl); // 2nd: scale
	return rdist(rng);
}

inline double ber_rand(double prob, boost::random::mt19937& rng) {
	boost::random::bernoulli_distribution<> rdist(prob); // Bernoulli supported -> use this instead of binomial
	return rdist(rng) * 1.0; // change to int later: now just use double to match Rf_rbinom
}

inline double unif_rand(double min, double max) {
	return Rf_runif(min, max);
}

inline double unif_rand(double min, double max, boost::random::mt19937& rng) {
	boost::random::uniform_real_distribution<> rdist(min, max);
	return rdist(rng);
}

inline double beta_rand(double s1, double s2) {
	return Rf_rbeta(s1, s2);
}

inline double beta_rand(double s1, double s2, boost::random::mt19937& rng) {
	boost::random::beta_distribution<> rdist(s1, s2);
	return rdist(rng);
}

#endif
