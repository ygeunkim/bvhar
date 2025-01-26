#ifndef BVHARSPDLOG_H
#define BVHARSPDLOG_H

#ifdef USE_RCPP

#include <RcppThread/Rcout.hpp>
#include <RcppSpdlog>
// #include <spdlog/pattern_formatter.h>
#include "omp.h"

namespace bvhar {
namespace sinks {

// Replace Rcpp::Rcout with RcppThread::Rcout in r_sink class
template<typename Mutex>
class bvhar_sink : public spdlog::sinks::r_sink<Mutex> {
protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
  #ifdef SPDLOG_USE_STD_FORMAT
    RcppThread::Rcout << formatted;
  #else
    RcppThread::Rcout << fmt::to_string(formatted);
  #endif
  }

  void flush_() override {
    RcppThread::Rcout << std::flush;
  }
};

using bvhar_sink_mt = bvhar_sink<std::mutex>;

} // namespace sinks

template<typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<spdlog::logger> bvhar_sink_mt(const std::string &logger_name) {
  return Factory::template create<sinks::bvhar_sink_mt>(logger_name);
}

} // namespace bvhar

	// #define Rcout RcppThreadRcout
	// namespace Rcpp {

	// 	static RcppThread::RPrinter RcppThreadRcout = RcppThread::RPrinter();

	// } // namespace Rcpp

	// #include <RcppSpdlog>

	// #define SPDLOG_SINK_MT(value) spdlog::r_sink_mt(value)
	#define SPDLOG_SINK_MT(value) bvhar::bvhar_sink_mt(value)

#else

	#include <spdlog/spdlog.h>
	#include <spdlog/sinks/stdout_sinks.h>

	#define SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

#endif // USE_RCPP

#endif // BVHARSPDLOG_H