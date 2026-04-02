/**
 * @file commoncpp.h
 * @author Young Geun Kim (ygeunkimstat@gmail.com)
 * @brief Rcpp and Pybind11 alternatives in C++
 */
#ifndef BVHAR_CORE_COMMONCPP_H
#define BVHAR_CORE_COMMONCPP_H

#include <cstdio>
#include <map>
#include <string>
#include <any>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

/**
 * @brief Works like Rcpp's list.
 * 
 */
using BvharList = std::map<std::string, std::any>;

template <typename T, typename = void>
struct bvhar_has_eval : std::false_type {};

template <typename T>
struct bvhar_has_eval<T, std::void_t<decltype(std::declval<T>().eval())>> : std::true_type {};

template <typename T>
inline constexpr bool bvhar_has_eval_v = bvhar_has_eval<T>::value;

/**
 * @brief Works like Rcpp's Named.
 * 
 */
struct BvharNamed {
  std::string name;
  explicit BvharNamed(const std::string& n) : name(n) {}
  
  template <typename T>
  std::pair<const std::string, std::any> operator=(T&& val) {
		if constexpr (bvhar_has_eval_v<T>) {
      return {name, val.eval()};
    }
    return {name, std::forward<T>(val)};
  }
};

/**
 * @brief Works like Rcpp's list create
 */
template <typename... Args>
BvharList create_bvhar_list(Args&&... args) {
  return BvharList{ std::forward<Args>(args)... };
}

inline void stop_fmt(const std::string& msg) {
	throw std::runtime_error(msg);
}

inline void stop_fmt(const char* msg) {
  throw std::runtime_error(msg);
}

template <typename... Args>
inline void stop_fmt(const char* fmt, Args... args) {
	// Use .c_str() instead of passing std::string
	int n = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
	if (n < 0) {
		throw std::runtime_error("BVHAR_STOP formatting failed");
	}
	std::string buf(static_cast<size_t>(n), '\0');
	std::snprintf(buf.data(), static_cast<size_t>(n) + 1, fmt, std::forward<Args>(args)...);
	throw std::runtime_error(buf);
}

#endif // BVHAR_CORE_COMMONCPP_H