#ifndef BVHAR_CORE_SPDLOG_H
#define BVHAR_CORE_SPDLOG_H

#include "./commondefs.h"
#include "./omp.h"

#ifdef BVHAR_USE_RCPP

// #include <RcppThread/Rcout.hpp>
#include <RcppSpdlog>
// #include <spdlog/pattern_formatter.h>
// #include "omp.h"

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

	// #define BVHAR_SPDLOG_SINK_MT(value) spdlog::r_sink_mt(value)
	#define BVHAR_SPDLOG_SINK_MT(value) bvhar::bvhar_sink_mt(value)

#else

	#include <spdlog/spdlog.h>
	#include <spdlog/sinks/stdout_sinks.h>

	#define BVHAR_SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

#endif // BVHAR_USE_RCPP

// Debug base classes by defining `BVHAR_USE_BVHAR_DEBUG`
#ifdef BVHAR_USE_BVHAR_DEBUG

#ifndef BVHAR_USE_SPDLOG
	#define BVHAR_USE_SPDLOG
#endif // BVHAR_USE_SPDLOG

#define BVHAR_DEBUG_LOGGER(value) \
	([](const std::string& log_name) -> std::shared_ptr<spdlog::logger> { \
		auto temp_logger = spdlog::get(log_name); \
		return temp_logger ? temp_logger : BVHAR_SPDLOG_SINK_MT(log_name); \
	})(value)
#define BVHAR_INIT_DEBUG(logger) logger->set_level(spdlog::level::debug)
#define BVHAR_DEBUG_LOG(logger, ...) logger->debug(__VA_ARGS__)
#define BVHAR_DEBUG_DROP(value) spdlog::drop(value)

#else

#define BVHAR_DEBUG_LOGGER(value) std::shared_ptr<spdlog::logger>(nullptr)
#define BVHAR_INIT_DEBUG(logger) (void)0
#define BVHAR_DEBUG_LOG(logger, ...) (void)0
#define BVHAR_DEBUG_DROP(value) (void)0

#endif // BVHAR_USE_BVHAR_DEBUG

#endif // BVHAR_CORE_SPDLOG_H