#ifndef BVHAROMP_H
#define BVHAROMP_H

#ifdef _OPENMP
  #include <omp.h>
#else
// #define omp_get_num_threads()  1
// #define omp_get_thread_num()   0
// #define omp_get_max_threads()  1
#ifndef omp_get_thread_num
	#define omp_get_thread_num() 0
#endif
#ifndef omp_get_max_threads
	#define omp_get_max_threads() 1
#endif
// #define omp_get_thread_limit() 1
// #define omp_get_num_procs()    1
// #define omp_set_nested(a)
// #define omp_get_wtime()        0
#endif

#include <atomic>
#include <mutex>
#include <vector> // std::vector in source file
#include <memory> // std::unique_ptr in source file

#endif
