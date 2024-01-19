.onAttach <- function(libname, pkgname) {
  if (!interactive()) {
    return()
  }
  packageStartupMessage(check_omp())
}