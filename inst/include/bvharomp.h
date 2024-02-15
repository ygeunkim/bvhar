#ifndef BVHAROMP_H
#define BVHAROMP_H

#ifdef _OPENMP
  #include <omp.h>
#else
#ifndef omp_get_thread_num
	#define omp_get_thread_num() 0
#endif // omp_get_thread_num
#ifndef omp_get_max_threads
	#define omp_get_max_threads() 1
#endif // omp_get_max_threads
#endif // _OPENMP

#include <atomic>
#include <mutex>
#include <vector> // std::vector in source file
#include <memory> // std::unique_ptr in source file

#endif // BVHAROMP_H
